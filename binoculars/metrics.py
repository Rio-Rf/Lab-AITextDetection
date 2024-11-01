import numpy as np
import torch
import transformers
import torch.nn.functional as F

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
mse_loss_fn = torch.nn.MSELoss(reduction="none") # meanにすると別のバッジの値も合計してしまう
softmax_fn = torch.nn.Softmax(dim=-1)


def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        #直接交差エントロピーを求めているのでlog PPLのlogをとる必要はない
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

    return ppl

def tv_perplexity(encoding: transformers.BatchEncoding,
                                 logits: torch.Tensor,
                                 median: bool = False,
                                 temperature: float = 1.0):
    # ロジットとラベルのシフト
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    # シフトされたロジットとラベルに基づく確率分布を取得
    #log_probs = F.log_softmax(shifted_logits, dim=-1) #このlogは交差エントロピーの計算上のlogなので不要
    probs = F.softmax(shifted_logits, dim=-1)
    label_probs = F.one_hot(shifted_labels, num_classes=probs.size(-1)).float() #one-hotベクトルに変換

    # 全変動距離を計算
    tv_distance = 0.5 * (probs - label_probs).abs().sum(dim=-1)

    if median:
        cosine_sim_nan = tv_distance.masked_fill(~shifted_attention_mask.bool(), float("nan"))
        ppl = np.nanmedian(cosine_sim_nan.cpu().float().numpy(), axis=1)
    else:
        # attention_maskでマスクした後、平均類似度を求める
        ppl = ((tv_distance * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)).to("cpu").float().numpy()

    return ppl

def cos_perplexity(encoding: transformers.BatchEncoding,
                                 logits: torch.Tensor,
                                 median: bool = False,
                                 temperature: float = 1.0):
    # ロジットとラベルのシフト
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    # シフトされたロジットとラベルに基づく確率分布を取得
    #log_probs = F.log_softmax(shifted_logits, dim=-1) #このlogは交差エントロピーの計算上のlogなので不要
    probs = F.softmax(shifted_logits, dim=-1)
    label_probs = F.one_hot(shifted_labels, num_classes=probs.size(-1)).float() #one-hotベクトルに変換

    # コサイン類似度を計算
    cosine_sim = F.cosine_similarity(probs, label_probs, dim=-1)

    if median:
        cosine_sim_nan = cosine_sim.masked_fill(~shifted_attention_mask.bool(), float("nan"))
        ppl = np.nanmedian(cosine_sim_nan.cpu().float().numpy(), axis=1)
    else:
        # attention_maskでマスクした後、平均類似度を求める
        ppl = ((cosine_sim * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)).to("cpu").float().numpy()

    return ppl

def js_perplexity(encoding: transformers.BatchEncoding,
                                 logits: torch.Tensor,
                                 median: bool = False,
                                 temperature: float = 1.0):
    # ロジットとラベルのシフト
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    total_tokens_available = shifted_logits.shape[-2]
    batch_size = shifted_logits.shape[0]  # バッチサイズ

    # シフトされたロジットとラベルに基づく確率分布を取得
    #log_probs = F.log_softmax(shifted_logits, dim=-1) #このlogは交差エントロピーの計算上のlogなので不要
    probs = F.softmax(shifted_logits, dim=-1)
    label_probs = F.one_hot(shifted_labels, num_classes=probs.size(-1)).float() #one-hotベクトルに変換

    # 平均分布M = 0.5 * (P + Q)
    m_proba = 0.5 * (probs + label_probs)
    
    # KLダイバージェンスの計算
    kl_p_m = F.kl_div(m_proba.log(), probs, reduction="none")
    kl_q_m = F.kl_div(m_proba.log(), label_probs, reduction="none")
    
    # JSダイバージェンス = 0.5 * (KL(P || M) + KL(Q || M))
    js_div = 0.5 * (kl_p_m + kl_q_m).sum(dim=-1)
    # [batch_size, total_tokens_available] の形に変換
    js_div = js_div.view(batch_size, total_tokens_available) #効いた

    if median:
        cosine_sim_nan = cosine_sim.masked_fill(~shifted_attention_mask.bool(), float("nan"))
        ppl = np.nanmedian(cosine_sim_nan.cpu().float().numpy(), axis=1)
    else:
        # attention_maskでマスクした後、平均類似度を求める
        ppl = ((js_div * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)).to("cpu").float().numpy()

    return ppl

# 32000次元の確率分布ベクトル、これを128回求める、それを32個の文章に対して行う
def entropy(p_logits: torch.Tensor, # 「観測された」確率分布を表す
            q_logits: torch.Tensor, # 「予測された」確率分布を表す
            encoding: transformers.BatchEncoding, # encodings は、トークン化された入力データのバッチを表します
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    vocab_size = p_logits.shape[-1] # モデルが予測するトークンの種類の数 32000
    total_tokens_available = q_logits.shape[-2] # 利用可能なトークン数
    p_scores, q_scores = p_logits / temperature, q_logits / temperature # tempが高いと確率分布がより滑らかになり、低いと分布が尖る(確率が極端に寄る)

    p_proba = softmax_fn(p_scores).view(-1, vocab_size) # ソフトマックス関数(softmax_fn)を適用して、ロジット(未加工のスコア)を確率分布に変換。[4096, 32000]

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    # ceの計算で自動的にソフトマックスが適用されるのでqにはsoftmaxを適用しない
    q_scores = q_scores.view(-1, vocab_size) #view()でわざわざ整形している [4096, 32000]

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available) # ce_loss_fn でクロスエントロピー損失を計算し、q_scores（予測された確率分布）と p_proba（観測された確率分布）との間の差異を測る。
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    #デバッグ用
    # print("ce shape:", ce.shape) ->([32, 128])
    # print("padding_mask shape:", padding_mask.shape) ->([32, 128])

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        # sum(1)で行方向(各バッチ=1つの文章)での合計を計算、
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy()) # 各トークンごとのクロスエントロピーの合計を、非パディング部分のトークン数で割って平均を算出しています。これにより、各入力に対してクロスエントロピーの平均値が得られます。
    # print("agg_ce shape:", agg_ce.shape) -> (32,)
    return agg_ce

def mse(p_logits: torch.Tensor,  # 観測された確率分布
                q_logits: torch.Tensor,  # 予測された確率分布
                encoding: transformers.BatchEncoding,  # トークン化された入力データのバッチ
                pad_token_id: int,
                median: bool = False,
                sample_p: bool = False,
                temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]  # モデルが予測するトークンの種類の数 32000
    total_tokens_available = q_logits.shape[-2]  # 利用可能なトークン数
    p_scores, q_scores = p_logits / temperature, q_logits / temperature  # 温度スケーリング [32, 128, 32000]

    p_proba = softmax_fn(p_scores).view(-1, vocab_size) # 観測された確率分布に変換 [4096, 32000]
    q_proba = softmax_fn(q_scores).view(-1, vocab_size) # 予測された確率分布に変換、誤差を出すためにp_probaと同じ形にした [4096, 32000]

    # 平均二乗誤差を計算
    mse = mse_loss_fn(q_proba, p_proba).view(-1, total_tokens_available) # [1024000, 128]
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)  # パディング部分を無視するためのマスク, 多分サイズは固定
    
    print("p_logits shape:", p_logits.shape)
    print("q_logits shape:", q_logits.shape)
    print("p_proba shape:", p_proba.shape)
    print("q_proba shape:", q_proba.shape)
    print("mse shape:", mse.shape)
    print("padding_mask shape:", padding_mask.shape)

    if median:
        mse_nan = mse.masked_fill(~padding_mask.bool(), float("nan"))
        agg_mse = np.nanmedian(mse_nan.cpu().float().numpy(), 1)
    else:
        # 平均二乗誤差を計算し、非パディング部分のトークン数で割って平均を求める
        agg_mse = (((mse * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy()) # (32,)
        print("agg_mse shape:", agg_mse.shape)
    return agg_mse

def kl_divergence(p_logits: torch.Tensor,  # 観測された確率分布
                               q_logits: torch.Tensor,  # 予測された確率分布
                               encoding: transformers.BatchEncoding,  # トークン化された入力データのバッチ
                               pad_token_id: int,
                               median: bool = False,
                               sample_p: bool = False,
                               temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]  # モデルが予測するトークンの種類の数
    batch_size = p_logits.shape[0]  # バッチサイズ
    seq_len = p_logits.shape[1]  # シーケンス長
    total_tokens_available = q_logits.shape[-2] # 利用可能なトークン数
    p_scores, q_scores = p_logits / temperature, q_logits / temperature  # 温度スケーリング

    # 観測されたスコアにソフトマックスを適用して確率分布に変換
    p_proba = F.softmax(p_scores, dim=-1).view(-1, vocab_size)  # [batch_size, seq_len, vocab_size]
    q_proba = F.log_softmax(q_scores, dim=-1).view(-1, vocab_size)  # log_softmax でKLダイバージェンスを計算

    # KLダイバージェンスを計算
    kl_div = F.kl_div(q_proba, p_proba, reduction='none') # [4096, 32000]

    # パディングマスクを作成し、[batch_size, seq_len] の形状をフラット化
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)
    padding_mask = padding_mask.view(-1)  # フラット化して [batch_size * seq_len] -> [4096]

# 各トークンのKLダイバージェンスを平均化

    # デバッグ用の形状確認
    #print("p_proba", p_proba.shape)  # -> [4096, 32000]
    #print("q_proba", q_proba.shape)  # -> [4096, 32000]
    #print("kl_div shape:", kl_div.shape)  # -> [4096, 32000]
    #print("padding_mask shape:", padding_mask.shape) # [4096]

    # 各トークンのKLダイバージェンスを平均化
    if median:
        kl_nan = kl_div.masked_fill(~padding_mask.bool().unsqueeze(-1), float("nan"))  # マスクを適用
        agg_kl = np.nanmedian(kl_nan.cpu().float().numpy(), axis=1)
    else:
        # パディングを考慮して平均KLダイバージェンスを計算
        kl_div_sum = (kl_div * padding_mask.unsqueeze(-1)).sum(1)  # [batch_size, seq_len] 方向に平均化
        non_pad_token_count = padding_mask.sum() #(1)を()にしたらエラーが出なくなった
        agg_kl = (kl_div_sum / non_pad_token_count).to("cpu").float().numpy()
    # バッチごとに集約して[32]に戻す
    agg_kl = agg_kl.reshape(batch_size, seq_len).mean(axis=1)  # [32]

    return agg_kl

#スコア8〜15程度の値を持つ
def total_variation_distance(p_logits: torch.Tensor,  # 観測された確率分布
                             q_logits: torch.Tensor,  # 予測された確率分布
                             encoding: transformers.BatchEncoding,  # トークン化された入力データのバッチ
                             pad_token_id: int,
                             median: bool = False,
                             sample_p: bool = False,
                             temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]  # モデルが予測するトークンの種類の数
    total_tokens_available = q_logits.shape[-2]  # 利用可能なトークン数
    p_scores, q_scores = p_logits / temperature, q_logits / temperature  # 温度スケーリング

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)  # ソフトマックス関数を適用して確率分布に変換
    q_proba = softmax_fn(q_scores).view(-1, vocab_size)  # 予測された確率分布

    # 総変動距離を計算
    tv_distance = 0.5 * (p_proba - q_proba).abs().sum(dim=-1).view(-1, total_tokens_available)  # [バッチサイズ * シーケンス長]
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    # デバッグ用の形状確認
    #print("p_proba", p_proba.shape)
    #print("q_proba", q_proba.shape)
    #print("tv_distance:", tv_distance.shape) #-> ([32, 128]) ここさえこのサイズならcsと同様な実装ができる
    #print("padding_mask shape:", padding_mask.shape) #-> ([32, 128])

    if median:
        tv_nan = tv_distance.masked_fill(~padding_mask.bool(), float("nan"))  # マスクを適用
        agg_tv = np.nanmedian(tv_nan.cpu().float().numpy(), axis=1)
    else:
        # パディングを考慮して平均総変動距離を計算
        tv_distance_sum = (tv_distance * padding_mask).sum(1)  # パディングを考慮して合計
        non_pad_token_count = padding_mask.sum(1)
        agg_tv = (tv_distance_sum / non_pad_token_count).to("cpu").float().numpy()  # 各バッチごとの平均
        # agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())
    return agg_tv

def cosine_similarity(p_logits: torch.Tensor, 
                      q_logits: torch.Tensor, 
                      encoding: transformers.BatchEncoding, 
                      pad_token_id: int,
                      median: bool = False,
                      sample_p: bool = False,
                      temperature: float = 1.0):
    vocab_size = p_logits.shape[-1] # モデルが予測するトークンの種類の数 32000
    total_tokens_available = q_logits.shape[-2] # 利用可能なトークン数
    p_scores, q_scores = p_logits / temperature, q_logits / temperature # tempが高いと確率分布がより滑らかになり、低いと分布が尖る(確率が極端に寄る)

    p_proba = softmax_fn(p_scores).view(-1, vocab_size) # ソフトマックス関数を適用して、ロジットを確率分布に変換。[4096, 32000]

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size) # [4096, 32000]

    # コサイン類似度の計算
    cosine_sim = F.cosine_similarity(p_proba, q_scores, dim=-1).view(-1, total_tokens_available) # トークンごとのコサイン類似度

    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    # デバッグ用
    # print("cosine_sim shape:", cosine_sim.shape) -> ([32, 128])
    # print("padding_mask shape:", padding_mask.shape) -> ([32, 128])

    if median:
        cosine_sim_nan = cosine_sim.masked_fill(~padding_mask.bool(), float("nan"))
        agg_cosine_sim = np.nanmedian(cosine_sim_nan.cpu().float().numpy(), 1)
    else:
        agg_cosine_sim = (((cosine_sim * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy()) # 各トークンごとのコサイン類似度の平均値を取得
    # print("agg_cosine_sim shape:", agg_cosine_sim.shape) -> (32,)
    return agg_cosine_sim

def js_divergence(p_logits: torch.Tensor,
                  q_logits: torch.Tensor,
                  encoding: transformers.BatchEncoding,
                  pad_token_id: int,
                  median: bool = False,
                  sample_p: bool = False,
                  temperature: float = 1.0):
    
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature
    batch_size = p_logits.shape[0]  # バッチサイズ

    # ソフトマックス関数を使って確率分布に変換
    p_proba = softmax_fn(p_scores)#.view(-1, vocab_size) 効いた
    q_proba = softmax_fn(q_scores)#.view(-1, vocab_size) 効いた

    if sample_p:
        p_proba = torch.multinomial(p_proba, replacement=True, num_samples=1).view(-1, vocab_size)

    # 平均分布M = 0.5 * (P + Q)
    m_proba = 0.5 * (p_proba + q_proba)
    
    # KLダイバージェンスの計算
    kl_p_m = F.kl_div(m_proba.log(), p_proba, reduction="none")
    kl_q_m = F.kl_div(m_proba.log(), q_proba, reduction="none")
    
    # JSダイバージェンス = 0.5 * (KL(P || M) + KL(Q || M))
    js_div = 0.5 * (kl_p_m + kl_q_m).sum(dim=-1)
    # [batch_size, total_tokens_available] の形に変換
    js_div = js_div.view(batch_size, total_tokens_available) #効いた
    
    # padding_maskでパディングを無視
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)
    
    # デバッグ用の形状確認
    #print("p_proba", p_proba.shape)
    #print("q_proba", q_proba.shape)
    #print("js_div", js_div.shape) #-> ([32, 128]) ここさえこのサイズならcsと同様な実装ができる
    #print("padding_mask shape:", padding_mask.shape) #-> ([32, 128])

    # JSダイバージェンスの集約（パディングを除外）
    if median:
        js_div_nan = js_div.masked_fill(~padding_mask.bool(), float("nan"))
        agg_js_div = np.nanmedian(js_div_nan.cpu().float().numpy(), 1)
    else:
        agg_js_div = (((js_div * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())
    
    return agg_js_div

def focal_loss(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0,
            alpha: float = 1.0,
            gamma: float = 2.0):
    vocab_size = p_logits.shape[-1]  # モデルが予測するトークンの種類の数 32000
    total_tokens_available = q_logits.shape[-2]  # 利用可能なトークン数
    p_scores, q_scores = p_logits / temperature, q_logits / temperature  # tempが高いと確率分布が滑らかになり、低いと尖る

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)  # ソフトマックス関数でロジットを確率分布に変換
    q_scores = q_scores.view(-1, vocab_size)  # 予測分布（q_logits）を適用する形に整形

    # 焦点損失（Focal Loss）を計算
    log_q_probs = F.log_softmax(q_scores, dim=-1)  # ソフトマックスを適用して対数を取る
    focal_weights = alpha * ((1 - torch.exp(log_q_probs)) ** gamma)  # 焦点損失の重み計算
    focal_loss = -focal_weights * p_proba * log_q_probs  # 焦点損失を適用
    focal_loss = focal_loss.sum(dim=-1).view(-1, total_tokens_available)  # バッチサイズとトークン数に戻す

    # パディングマスクの適用
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        focal_loss_nan = focal_loss.masked_fill(~padding_mask.bool(), float("nan"))
        agg_loss = np.nanmedian(focal_loss_nan.cpu().float().numpy(), 1)
    else:
        # 非パディング部分のみで平均を取る
        agg_loss = ((focal_loss * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy()
    
    return agg_loss