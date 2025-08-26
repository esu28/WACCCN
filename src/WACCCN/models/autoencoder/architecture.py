# autoencoder 
# Multi-view CCC-aware autoencoder (TF/Keras) with recon + Laplacian + InfoNCE + DEC.

import os, json, numpy as np, tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, TimeDistributed, LayerNormalization
from tensorflow.keras.initializers import HeNormal, GlorotUniform

EPS = 1e-8


def zscore(X):
    m = X.mean(0, keepdims=True)
    s = X.std(0, keepdims=True) + EPS
    return ((X - m) / s).astype(np.float32)


def prepare_ccc(CCC):
    CCC = np.nan_to_num(CCC, nan=0.0, posinf=0.0, neginf=0.0)
    CCC = np.maximum(CCC, 0.0).astype(np.float32)
    n = CCC.shape[0]
    idx = np.arange(n)
    CCC[idx[:, None], idx[None, :], :] = 0.0
    p95 = np.quantile(CCC, 0.95, axis=(0, 1), keepdims=True) + EPS
    CCC = np.clip(CCC / p95, 0.0, 1.0).astype(np.float32)
    return CCC


def topk_mask(W, k):
    k = int(min(k, W.shape[1] - 1))
    idxs = np.argpartition(-W, kth=k, axis=1)[:, :k]
    M = np.zeros_like(W, dtype=np.float32)
    row = np.repeat(np.arange(W.shape[0]), k)
    M[row, idxs.ravel()] = 1.0
    np.fill_diagonal(M, 0.0)
    return M


def build_ae(input_dims, embed_d=32, dropout=0.15, seed=1):
    F_tf = input_dims["tf"]
    F_hist = input_dims["hist"]
    F_gene = input_dims["gene"]
    F_one = input_dims["onehot"]
    F_gdwt = input_dims["gdwt"]
    P = input_dims["P"]

    init_he = HeNormal(seed=seed)
    init_gl = GlorotUniform(seed=seed)

    tf_in = Input(shape=(None, F_tf), name="TF")
    hist_in = Input(shape=(None, F_hist), name="HISDWT")
    gene_in = Input(shape=(None, F_gene), name="GENE")
    one_in = Input(shape=(None, F_one), name="ONEHOT")
    gdwt_in = Input(shape=(None, F_gdwt), name="GDWT")
    ccc_in = Input(shape=(None, None, P), name="CCC")

    def view_mlp(units, name_prefix):
        def f(x):
            x = TimeDistributed(Dense(units, activation="gelu", kernel_initializer=init_he), name=f"{name_prefix}_d1")(x)
            x = TimeDistributed(LayerNormalization(), name=f"{name_prefix}_ln1")(x)
            x = TimeDistributed(Dropout(dropout), name=f"{name_prefix}_do1")(x)
            x = TimeDistributed(Dense(units, activation="gelu", kernel_initializer=init_he), name=f"{name_prefix}_d2")(x)
            x = TimeDistributed(LayerNormalization(), name=f"{name_prefix}_ln2")(x)
            return x
        return f

    class WeightedCCCEncode(tf.keras.layers.Layer):
        def __init__(self, num_pairs, **kw):
            super().__init__(**kw)
            self.num_pairs = num_pairs
        def build(self, _):
            self.scores = self.add_weight("ccc_scores", shape=(self.num_pairs,), initializer="zeros", dtype=tf.float32, trainable=True)
        def call(self, inputs):
            x, S = inputs  # x: (B,N,F), S: (B,N,N,P)
            gamma = tf.nn.softmax(self.scores)  # (P,)
            Sg = tf.tensordot(S, gamma, axes=([3], [0]))  # (B,N,N)
            return tf.einsum("bij,bjf->bif", Sg, x)

    tf_e = view_mlp(64, "tf")(tf_in)
    hist_e = view_mlp(64, "hist")(hist_in)
    gene_e = view_mlp(64, "gene")(gene_in)
    one_e = view_mlp(32, "one")(one_in)
    gdwt_e = view_mlp(64, "gdwt")(gdwt_in)

    xfeat = Concatenate(axis=-1, name="concat_views")([tf_e, hist_e, gene_e, one_e, gdwt_e])
    xfeat = TimeDistributed(LayerNormalization(), name="feat_ln")(xfeat)

    z_msg = WeightedCCCEncode(P, name="CCC_Agg")([xfeat, ccc_in])
    hcat = Concatenate(axis=-1, name="concat_msg")([xfeat, z_msg])

    z = TimeDistributed(Dense(128, activation="gelu", kernel_initializer=init_he), name="z_d1")(hcat)
    z = TimeDistributed(LayerNormalization(), name="z_ln1")(z)
    z = TimeDistributed(Dropout(0.15 if dropout is None else dropout), name="z_do1")(z)
    z = TimeDistributed(Dense(embed_d, activation=None, kernel_initializer=init_gl), name="z_proj")(z)
    z = LayerNormalization(axis=-1, name="Latent")(z)

    def dec_head(z, dim, name):
        h = TimeDistributed(Dense(128, activation="gelu", kernel_initializer=init_he), name=f"{name}_h")(z)
        h = TimeDistributed(LayerNormalization(), name=f"{name}_ln")(h)
        return TimeDistributed(Dense(dim, activation=None, kernel_initializer=init_gl), name=f"{name}_out")(h)

    y_tf = dec_head(z, F_tf, "dec_tf")
    y_hist = dec_head(z, F_hist, "dec_hist")
    y_gene = dec_head(z, F_gene, "dec_gene")
    y_one = dec_head(z, F_one, "dec_one")
    y_gdwt = dec_head(z, F_gdwt, "dec_gdwt")

    model = Model(
        inputs=[tf_in, hist_in, gene_in, one_in, gdwt_in, ccc_in],
        outputs=[y_tf, y_hist, y_gene, y_one, y_gdwt, z],
        name="AE_LeidenReady",
    )
    return model


def fit_ae(
    X_tf,
    X_hist,
    X_gene,
    X_one,
    X_gdwt,
    CCC,
    embed_d=32,
    clusters_k=30,
    epochs=40,
    patience=40,
    lr=7e-4,
    topk_pos=18,
    tau=0.12,
    lamb_recon=1.0,
    lamb_laplacian=0.08,
    lamb_info_nce=0.25,
    lamb_dec=0.15,
    dec_update_every=10,
    dropout=0.15,
    seed=1,
    zscore_gene=False,
    save_weights=None,
    save_latent=None,
    save_manifest=None,
):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    N = X_tf.shape[0]
    P = CCC.shape[2]

    X_tf = zscore(X_tf)
    X_hist = zscore(X_hist)
    if zscore_gene:
        X_gene = zscore(X_gene)
    X_gdwt = zscore(X_gdwt)

    CCC = prepare_ccc(CCC)

    W = CCC.mean(axis=2).astype(np.float32)
    np.fill_diagonal(W, 0.0)
    row_sum = W.sum(axis=1, keepdims=True) + EPS
    W_norm = (W / row_sum).astype(np.float32)
    POS_MASK = topk_mask(W, topk_pos).astype(np.float32)

    input_dims = {
        "tf": X_tf.shape[1],
        "hist": X_hist.shape[1],
        "gene": X_gene.shape[1],
        "onehot": X_one.shape[1],
        "gdwt": X_gdwt.shape[1],
        "P": P,
    }
    model = build_ae(input_dims, embed_d=embed_d, dropout=dropout, seed=seed)
    opt = tf.keras.optimizers.Adam(lr)

    centers = tf.Variable(tf.random.normal([clusters_k, embed_d], stddev=0.1, seed=seed), name="dec_centers")
    mse = tf.keras.losses.MeanSquaredError(reduction="mean")

    def recon_loss(y_true, y_pred):
        return mse(y_true, y_pred)

    def laplacian_loss(z, Wn):
        zi = z[0]
        z_nb = tf.matmul(Wn, zi)
        return tf.reduce_mean(tf.reduce_sum((zi - z_nb) ** 2, axis=1))

    def info_nce_loss(z, pos_mask, tau_):
        zi = tf.math.l2_normalize(z[0], axis=1)
        sim = tf.matmul(zi, zi, transpose_b=True) / tau_
        sim = sim - 1e9 * tf.eye(tf.shape(sim)[0], dtype=sim.dtype)
        logsumexp = tf.reduce_logsumexp(sim, axis=1, keepdims=True)
        log_prob = sim - logsumexp
        pos_count = tf.reduce_sum(pos_mask, axis=1) + EPS
        loss_i = -tf.reduce_sum(pos_mask * log_prob, axis=1) / pos_count
        return tf.reduce_mean(loss_i)

    def student_t_q(z):
        zi = z[0]
        dist2 = tf.reduce_sum((zi[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        q = 1.0 / (1.0 + dist2)
        return q / tf.reduce_sum(q, axis=1, keepdims=True)

    def target_p(q):
        f = q**2 / (tf.reduce_sum(q, axis=0, keepdims=True) + EPS)
        return f / (tf.reduce_sum(f, axis=1, keepdims=True) + EPS)

    Wn_tf = tf.constant(W_norm)
    POS_tf = tf.constant(POS_MASK)

    x_tf_b = X_tf[None, ...]
    x_hist_b = X_hist[None, ...]
    x_gene_b = X_gene[None, ...]
    x_one_b = X_one[None, ...]
    x_gdwt_b = X_gdwt[None, ...]
    ccc_b = CCC[None, ...]

    best_loss, wait = np.inf, 0
    q_cur = student_t_q(model([x_tf_b, x_hist_b, x_gene_b, x_one_b, x_gdwt_b, ccc_b])[-1])
    p_cur = target_p(q_cur)

    for epoch in range(1, epochs + 1):
        with tf.GradientTape() as tape:
            ytf, yhi, yge, yon, ygd, z = model([x_tf_b, x_hist_b, x_gene_b, x_one_b, x_gdwt_b, ccc_b], training=True)

            L_rec = (
                recon_loss(x_tf_b, ytf) / X_tf.shape[1]
                + recon_loss(x_hist_b, yhi) / X_hist.shape[1]
                + recon_loss(x_gene_b, yge) / X_gene.shape[1]
                + recon_loss(x_one_b, yon) / X_one.shape[1]
                + recon_loss(x_gdwt_b, ygd) / X_gdwt.shape[1]
            )

            L_lap = laplacian_loss(z, Wn_tf)
            L_nce = info_nce_loss(z, POS_tf, tau)
            q = student_t_q(z)
            L_dec = tf.reduce_sum(p_cur * (tf.math.log(p_cur + EPS) - tf.math.log(q + EPS))) / N

            L_total = lamb_recon * L_rec + lamb_laplacian * L_lap + lamb_info_nce * L_nce + lamb_dec * L_dec

        vars_all = model.trainable_variables + [centers]
        grads = tape.gradient(L_total, vars_all)
        opt.apply_gradients(zip(grads, vars_all))

        if epoch % dec_update_every == 0:
            z_eval = model([x_tf_b, x_hist_b, x_gene_b, x_one_b, x_gdwt_b, ccc_b], training=False)[-1]
            q_cur = student_t_q(z_eval)
            p_cur = target_p(q_cur)

        tot = float(L_total.numpy())
        if tot < best_loss - 1e-5:
            best_loss, wait = tot, 0
            z_best = model([x_tf_b, x_hist_b, x_gene_b, x_one_b, x_gdwt_b, ccc_b], training=False)[-1].numpy()[0]
            if save_weights:
                os.makedirs(os.path.dirname(save_weights), exist_ok=True)
                model.save_weights(save_weights)
        else:
            wait += 1

        if wait >= patience:
            break

    if "z_best" not in locals():
        z_best = model([x_tf_b, x_hist_b, x_gene_b, x_one_b, x_gdwt_b, ccc_b], training=False)[-1].numpy()[0]

    if save_latent:
        os.makedirs(os.path.dirname(save_latent), exist_ok=True)
        np.save(save_latent, z_best)

    if save_manifest:
        os.makedirs(os.path.dirname(save_manifest), exist_ok=True)
        with open(save_manifest, "w") as f:
            json.dump(
                {
                    "N": int(N),
                    "embed_dim": int(embed_d),
                    "clusters_k": int(clusters_k),
                    "loss_weights": {
                        "recon": float(lamb_recon),
                        "laplacian": float(lamb_laplacian),
                        "infonce": float(lamb_info_nce),
                        "dec": float(lamb_dec),
                    },
                    "topk_pos": int(topk_pos),
                    "tau": float(tau),
                    "epochs_done": int(epoch),
                    "best_total_loss": float(best_loss),
                    "paths": {"weights": save_weights, "latent": save_latent},
                },
                f,
                indent=2,
            )

    return z_best, model
