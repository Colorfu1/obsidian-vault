---
github_repo: Colorfu1|obsidian-vault
---
# Pi_0机器人文章分析

**User:** Anonymous  
**Created:** 5/17/2026 19:50:15  
**Updated:** 5/20/2026 16:09:26  
**Exported:** 5/20/2026 16:17:07  
**Link:** [https://chatgpt.com/c/6a09ab66-6cd0-83ec-85a3-69240e49f621](https://chatgpt.com/c/6a09ab66-6cd0-83ec-85a3-69240e49f621)  

## Response:
5/20/2026, 4:08:37 PM

下面是一份基于我们前面讨论整理的 **π0 / openpi / π0-FAST 知识总结**。

---

# π0 相关知识点总结

## 1. π0 这篇文章的核心 claim

π0 的核心主张不是“让机器人生成 high-level policy steps”，而是：

> 用预训练 VLM 提供视觉-语言语义能力，用 flow matching 直接生成连续动作，并通过大规模跨机器人数据训练出一个 general robot control foundation policy。

可以拆成三点：

```text
1. VLM backbone 提供视觉-语言理解能力
2. Action expert + flow matching 负责连续动作生成
3. 多机器人、多任务、大规模数据训练带来泛化能力
```

所以 π0 更像：

```text
语言条件下的低层连续动作生成器
```

而不是：

```text
显式语言 planner + action generator 的完整层级模型
```

---

## 2. π0 的输入和输出

π0 的输入并不只是 camera 和 language。完整来说，一条训练样本通常包含：

```text
输入:
    多视角 RGB 图像
    语言指令
    robot state / proprioception
        例如关节角、末端位姿、夹爪状态等
    noisy action chunk
        训练 flow matching 时使用

监督:
    expert future action chunk
```

输出是：

```text
future action chunk
```

例如：

```text
actions: [H, action_dim]
```

π0 论文里常用：

```text
H = 50
action_dim 最高 pad 到 18
```

---

## 3. π0 的整体结构

π0 可以理解成：

```text
PaliGemma VLM backbone
        +
Action Expert
        +
Flow Matching Head
```

数据流大概是：

```text
图像 + 语言指令
        ↓
PaliGemma / VLM backbone

robot state + noisy action chunk + timestep
        ↓
Action expert

二者通过 Transformer attention 交互
        ↓
预测 action velocity field
        ↓
ODE / Euler 积分
        ↓
continuous action chunk
```

在 openpi 的 `pi0.py` 中，原版 π0 的 suffix 输入包含 state token 和 noisy action tokens，最后只取 action horizon 对应的 suffix 输出，再投影回 action 维度。`pi0.py`

---

## 4. PaliGemma 是什么？

PaliGemma 是 Google 的视觉-语言模型，可以简单理解为：

```text
PaliGemma = SigLIP 图像编码器 + Gemma 语言模型
```

它原本用于：

```text
图像 + 文本 prompt → 文本回答
```

例如：

```text
输入: 图片 + "what is on the table?"
输出: "a red cup"
```

在 π0 里，PaliGemma 不是用来聊天，而是作为视觉-语言语义 backbone：

```text
负责:
    看懂图像
    理解语言指令
    对齐图像区域和语言语义

不直接负责:
    输出连续 robot action
```

π0 新增的 action expert 才负责把这些语义信息转成机器人动作。

---

## 5. π0 使用的是 flow matching，不是传统 action token CE

原版 π0 的训练目标是 flow matching。

在 openpi `pi0.py` 中，loss 逻辑可以概括为：

```python
noise = random_normal(actions.shape)
time = random_beta(...)
x_t = time * noise + (1 - time) * actions
u_t = noise - actions

v_t = model(x_t, time, observation)

loss = mean((v_t - u_t) ** 2)
```

也就是说：

```text
真实动作 actions
        +
噪声 noise
        ↓
构造中间状态 x_t
        ↓
模型预测 velocity v_t
        ↓
用 MSE 监督 v_t ≈ noise - actions
```

代码里原版 π0 的 `compute_loss()` 返回的是：

```python
jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

不是 cross entropy。`pi0.py`

---

## 6. Flow matching 中 velocity 的 shape

如果被建模的数据是：

```text
x: [H, W, C]
```

那么 flow matching 的输出 velocity 通常也是：

```text
v: [H, W, C]
```

因为 ODE 更新是：

```text
x_{t+dt} = x_t + dt * v_t
```

所以 `x_t` 和 `v_t` 必须 shape 一致。

在 π0 里：

```text
action chunk: [H, action_dim]
velocity:     [H, action_dim]
```

---

## 7. π0 是否生成 high-level policy steps？

我们的结论是：

> π0 v1 主模型没有明确展示“生成 high-level policy steps”的能力。

它可以接收语言指令，也可以接收外部 high-level VLM policy 生成的子任务语言，然后执行动作。

也就是：

```text
外部 high-level VLM policy:
    当前图像 + 总任务 → 子任务语言

π0:
    当前图像 + 子任务语言 + robot state → continuous action chunk
```

但这不等于：

```text
π0 内部自己生成:
    step 1: pick up cup
    step 2: move to bowl
    step 3: release cup
然后再生成 actions
```

论文示意图里没有画出 high-level policy steps 分支，是合理的，因为它不是 π0 v1 核心结构的一部分。

---

## 8. 论文里提到 Transfusion，是什么意思？

Transfusion 的思想是：

```text
离散 token:
    用 cross entropy / language modeling loss

连续数据:
    用 diffusion / flow-style loss
```

π0 借鉴的是这种“同一个 Transformer 框架里处理离散语言和连续动作”的思想。

但在原版 π0 中：

```text
language / image:
    作为条件输入

action:
    用 flow matching 建模
```

并不是说 π0 默认会用 CE 生成 high-level policy step 文本。

---

## 9. openpi 里的 CE loss 是什么？

openpi 代码里确实有 CE loss，但它在 `pi0_fast.py` 里，不是原版 `pi0.py`。

`pi0_fast.py` 的 `compute_loss()` 是标准 next-token CE：

```python
targets = one_hot(observation.tokenized_prompt[:, 1:])
logp = log_softmax(logits)
loss_mask = observation.token_loss_mask[:, 1:]
loss = -sum(targets * logp * loss_mask) / sum(loss_mask)
```

`pi0_fast.py`

但这个 CE 不是训练 high-level policy steps，而是训练 **action tokens**。

---

## 10. action的生成是单独的flowmatching的一个transformer生成的, 那么从代码上看, 作为cross attention的图像/语言输入是如何作为flow matching的cross attention对象的? 

我会直接按代码调用链拆：`embed_prefix()` 生成图像/语言 prefix，`embed_suffix()` 生成 state/action/time suffix，`make_attn_mask()` 控制 prefix/suffix 互看关系，`gemma.Module` 里把两个 expert 的 QKV 拼起来做 joint attention。

目前关键证据已经抓到了：原版 π0 训练时是 prefix+suffix 一次 forward；推理时为了效率先把 prefix KV cache 固定，再让每个 flow step 的 action suffix 读这个 KV cache。下面再抓一下 Gemma 里 joint attention 的实现。

可以。我们直接按 openpi 代码把这句话落到实现上：

```text
(I1, I2, L1, L2) hidden prefix
    ↘
      joint attention
    ↗
(noisy action tokens + timestep) suffix
    → velocity
```

在 openpi 里，这个过程主要发生在两个文件：

```text
src/openpi/models/pi0.py
src/openpi/models/gemma.py
```

---

### 10.1. 模型初始化：PaliGemma expert + action expert

`Pi0.__init__()` 里构造了两个 expert config：

```python
paligemma_config = _gemma.get_config(config.paligemma_variant)
action_expert_config = _gemma.get_config(config.action_expert_variant)
```

然后把两个 config 一起传给同一个 `_gemma.Module`：

```python
_gemma.Module(
    configs=[paligemma_config, action_expert_config],
    embed_dtype=config.dtype,
    adarms=config.pi05,
)
```

这说明它不是两个完全独立的 Transformer，而是一个 **support multiple experts 的 Gemma Module**：第 0 个 expert 对应 PaliGemma/Gemma，第 1 个 expert 对应 action expert。`pi0.py`

同时还有：

```python
self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
```

也就是 action branch 自己有输入投影、time/action 融合 MLP、输出投影。`pi0.py`

---

### 10.2. `embed_prefix()`：图像/语言变成 prefix tokens

代码里 prefix 是这样生成的：

```python
for name in obs.images:
    image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
    tokens.append(image_tokens)
```

这一步对应你的：

```text
I1, I2 → image tokens
```

这里的 `self.PaliGemma.img` 是 SigLIP 图像 encoder，它输出一串 image tokens。

然后语言 prompt 被 embed：

```python
tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
tokens.append(tokenized_inputs)
```

这一步对应：

```text
L1, L2 → language token embeddings
```

最后：

```python
tokens = jnp.concatenate(tokens, axis=1)
```

得到完整 prefix：

```text
prefix_tokens = [image_tokens..., language_tokens...]
```

而且 prefix 的 `ar_mask` 都是 `False`：

```python
# image tokens attend to each other
ar_mask += [False] * image_tokens.shape[1]

# full attention between image and language inputs
ar_mask += [False] * tokenized_inputs.shape[1]
```

这表示图像/语言 prefix 之间可以 full attention。`pi0.py`

所以你写的：

```text
(I1, I2, L1, L2) hidden prefix
```

在代码里基本就是：

```python
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
```

---

### 10.3. `embed_suffix()`：state / noisy action / timestep 变成 suffix tokens

原版 π0 里 suffix 首先加入 state token：

```python
state_token = self.state_proj(obs.state)[:, None, :]
tokens.append(state_token)
```

然后 noisy actions 被投影成 action tokens：

```python
action_tokens = self.action_in_proj(noisy_actions)
```

timestep 先做 sin/cos embedding：

```python
time_emb = posemb_sincos(
    timestep,
    self.action_in_proj.out_features,
    min_period=4e-3,
    max_period=4.0,
)
```

然后把 time embedding repeat 到 action horizon 上，并和 action token 拼接：

```python
time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
action_time_tokens = self.action_time_mlp_in(action_time_tokens)
action_time_tokens = nnx.swish(action_time_tokens)
action_time_tokens = self.action_time_mlp_out(action_time_tokens)
action_expert_tokens = action_time_tokens
```

最后 suffix 是：

```text
suffix_tokens = [state_token, action_time_token_1, ..., action_time_token_H]
```

代码里也对应：

```python
tokens.append(action_expert_tokens)
tokens = jnp.concatenate(tokens, axis=1)
return tokens, input_mask, ar_mask, adarms_cond
```

`pi0.py` `pi0.py`

所以你写的：

```text
(noisy action tokens + timestep) suffix
```

在代码里就是：

```python
suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
    observation, x_t, time
)
```

---

### 10.4. 训练时：prefix + suffix 一次性 joint forward

在 `compute_loss()` 里，先构造 flow matching 的 noisy action：

```python
noise = jax.random.normal(noise_rng, actions.shape)
time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
x_t = time_expanded * noise + (1 - time_expanded) * actions
u_t = noise - actions
```

然后生成 prefix 和 suffix：

```python
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
    observation, x_t, time
)
```

再把 mask 拼起来：

```python
input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
attn_mask = make_attn_mask(input_mask, ar_mask)
positions = jnp.cumsum(input_mask, axis=1) - 1
```

然后关键调用：

```python
(prefix_out, suffix_out), _ = self.PaliGemma.llm(
    [prefix_tokens, suffix_tokens],
    mask=attn_mask,
    positions=positions,
    adarms_cond=[None, adarms_cond],
)
```

这里 `[prefix_tokens, suffix_tokens]` 就是两个 expert 的输入列表：

```text
expert 0 input = prefix_tokens
expert 1 input = suffix_tokens
```

然后只取 suffix 的 action 部分输出：

```python
v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
```

最终 loss：

```python
return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

所以训练时完整路径就是：

```text
image/language prefix tokens
        +
state/noisy-action/time suffix tokens
        ↓
self.PaliGemma.llm([prefix_tokens, suffix_tokens])
        ↓
suffix_out 最后 H 个 token
        ↓
action_out_proj
        ↓
velocity v_t
        ↓
MSE(v_t, noise - actions)
```

`pi0.py`

这就是代码层面的：

```text
(I1, I2, L1, L2) hidden prefix
    ↘
      joint attention
    ↗
(noisy action tokens + timestep) suffix
    → velocity
```

---

### 10.5. `make_attn_mask()`：为什么 action suffix 能看 prefix？

`make_attn_mask()` 的注释非常关键。它说 `mask_ar` 可以构造 prefix-LM attention，例如：

```text
[[0 0 0 1 1 1]]
```

前 3 个 token 可以互相 attend，后 3 个 token 是 causal attention，并可以看前面的 prefix。`pi0.py`

π0 里 prefix 的 `ar_mask` 是 `False`，suffix 开头 state 是 `True`，后续 action tokens 是：

```python
ar_mask += [True] + ([False] * (self.action_horizon - 1))
```

这表示：

```text
prefix image/language:
    一个 full-attention block

suffix state + action horizon:
    另一个 block

suffix block 可以 attend 到 prefix block
prefix block 不 attend 到 suffix block
```

代码注释也写得很明确：

```python
# image/language/state inputs do not attend to action tokens
```

也就是说不是 prefix 去看 action，而是 action suffix 读取 prefix 作为条件。`pi0.py`

---

### 10.6. 推理时：prefix 先 KV cache，suffix 每个 flow step 读 prefix

推理 `sample_actions()` 里，为了效率，prefix 只算一次：

```python
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
positions = jnp.cumsum(prefix_mask, axis=1) - 1
_, kv_cache = self.PaliGemma.llm(
    [prefix_tokens, None],
    mask=prefix_attn_mask,
    positions=positions,
)
```

这里 `[prefix_tokens, None]` 表示只跑 PaliGemma/prefix expert，把 image/language 的 K/V cache 存下来。`pi0.py`

然后每个 flow step 里重新构造 suffix：

```python
suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
    observation, x_t, jnp.broadcast_to(time, batch_size)
)
```

构造 suffix 自己的 attention mask：

```python
suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
```

再构造 suffix 对 prefix 的 attention mask：

```python
prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
```

注释直接说明：

```python
# prefix_attn_mask is shape (b, suffix_len, prefix_len)
# indicating how the suffix tokens can attend to the prefix tokens
```

然后合并：

```python
full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
```

注释也明确说：

```python
# suffix tokens ... can attend to the full prefix + suffix sequence
# which generates the keys and values
```

最后调用：

```python
(prefix_out, suffix_out), _ = self.PaliGemma.llm(
    [None, suffix_tokens],
    mask=full_attn_mask,
    positions=positions,
    kv_cache=kv_cache,
    adarms_cond=[None, adarms_cond],
)
assert prefix_out is None
v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
```

所以推理时就是：

```text
prefix image/language:
    先算一次，存 KV cache

每个 flow step:
    noisy action x_t + time → suffix tokens
    suffix tokens 的 Q attend 到:
        prefix KV cache
        suffix 自己的 K/V
    得到 suffix_out
    投影成 velocity
```

`pi0.py`

这就非常接近 cross-attention 的效果，但实现上是 KV-cache + masked self-attention。

---

### 10.7. `gemma.py`：joint attention 是怎么发生的？

`gemma.Module` 的注释说：

```python
"""Transformer model, supporting a mixture of different weights for different tokens."""
```

也就是说，它支持不同 token 类型用不同 expert 权重。`gemma.py`

在 `Attention.__call__()` 里，它对每个 expert 的输入 `xs` 分别算 Q/K/V：

```python
for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
    ...
    q = ...
    k, v = ...
    qkvs.append((q, k, v))
```

然后非常关键的一句：

```python
q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
```

这表示 prefix expert 和 action expert 产生的 Q/K/V 会在 **token 维度** 拼起来，进入同一个 attention 计算。`gemma.py`

接下来正常算 attention：

```python
logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, ...)
masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
```

这里的 `attn_mask` 就决定 suffix tokens 能看哪些 prefix/suffix keys。`gemma.py`

算完后，它再按 expert token 段切回去：

```python
start = 0
for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
    if x is not None:
        end = start + x.shape[1]
        out.append(... encoded[:, start:end] ...)
        start = end
```

所以流程是：

```text
prefix expert tokens:
    用 PaliGemma expert 权重产生 Q/K/V

action expert tokens:
    用 action expert 权重产生 Q/K/V

然后:
    Q/K/V 沿 token 维度 concat
    统一做 attention
    attention 后再切回 prefix_out / suffix_out
```

这就是我说的 **joint attention**。不是先 VLM 生成文本，不是两个模块串行；而是两个 token stream 在 attention 里交互。

---

### 10.8. 整体伪代码

按 openpi 的代码，可以等价写成：

```python
# 1. prefix: image + language
image_tokens = SigLIP(images)
language_tokens = GemmaEmbed(tokenized_prompt)
prefix_tokens = concat([image_tokens, language_tokens], dim=seq)

# 2. suffix: state + noisy action + time
state_token = Linear(state)
action_tokens = Linear(noisy_actions)
time_tokens = repeat(sincos(t), H)
action_time_tokens = MLP(concat([action_tokens, time_tokens], dim=-1))
suffix_tokens = concat([state_token, action_time_tokens], dim=seq)

# 3. attention mask
attn_mask = make_prefix_suffix_mask(prefix_mask, suffix_mask)

# 4. joint expert transformer
prefix_out, suffix_out = GemmaModule(
    [prefix_tokens, suffix_tokens],
    mask=attn_mask,
)

# 5. action velocity
v_t = Linear(suffix_out[:, -H:])
loss = mse(v_t, noise - actions)
```

推理时只是把第 1 步 prefix 的 KV cache 存下来，后续每个 flow step 只重新跑 suffix：

```python
kv_cache = GemmaModule([prefix_tokens, None])

for t in flow_steps:
    suffix_tokens = embed_suffix(x_t, t)
    suffix_out = GemmaModule(
        [None, suffix_tokens],
        kv_cache=kv_cache,
        mask=full_attn_mask,
    )
    v_t = action_out_proj(suffix_out[:, -H:])
    x_t = x_t + dt * v_t
```

---

## 最终结论

openpi 代码证明，π0 的 action flow 不是这样：

```text
image/language → VLM 自回归生成文字 → EOF → action transformer
```

而是这样：

```text
image tokens + language tokens
        ↓
prefix hidden tokens / prefix KV

state + noisy action + timestep
        ↓
suffix action-expert tokens

prefix + suffix
        ↓
same Gemma Module, mixture-of-experts weights
masked joint self-attention
        ↓
suffix_out
        ↓
action_out_proj
        ↓
velocity [H, action_dim]
```

所以你那句可以更准确地写成：

```text
image/language prefix tokens
    provide K/V context through masked joint attention

noisy action + timestep suffix tokens
    provide Q and also suffix K/V

suffix hidden states
    → linear projection
    → flow velocity
```



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)

## 11. π0-FAST 是什么？

π0-FAST 和原版 π0 不一样。

原版 π0：

```text
continuous action
        ↓
flow matching
        ↓
continuous action output
```

π0-FAST：

```text
continuous action
        ↓
FAST action tokenizer
        ↓
discrete action tokens
        ↓
LLM-style next-token prediction
        ↓
decode 回 continuous action
```

所以 π0-FAST 把动作变成 LLM 可以预测的 token 序列。

---

## 12. π0-FAST 的训练序列长什么样？

在 `FASTTokenizer.tokenize()` 里，训练序列大概是：

```text
Task: <任务语言>, State: <离散化状态>;
Action: <FAST action tokens> |
```

代码里先构造 prefix：

```python
prefix = f"Task: {cleaned_text}, State: {state_str};\n"
```

然后如果有 actions，会把连续动作 token 化：

```python
action_tokens = self._fast_tokenizer(actions[None])[0]
action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)
```

再拼成 postfix：

```python
postfix_tokens = (
    self._paligemma_tokenizer.encode("Action: ")
    + action_tokens_in_pg.tolist()
    + self._paligemma_tokenizer.encode("|", add_eos=True)
)
```

并且 loss mask 是：

```python
loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)
```

所以 CE 只算 `"Action:"` 后面的 action token 区域。`tokenizer.py`

---

## 13. `_paligemma_tokenizer` 是什么？

`_paligemma_tokenizer` 不是神经网络，也不是 encoder-decoder 模型。

它只是一个 SentencePiece tokenizer：

```text
encode:
    text → token ids

decode:
    token ids → text
```

所以不能用 “decoder-only / encoder-decoder” 来描述 tokenizer。  
“decoder-only LLM” 说的是 Transformer 模型结构，不是 tokenizer 的 decode 函数。

---

## 14. π0-FAST 里的 `tokens` 是什么？

在：

```python
decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())
```

这里的 `tokens` 是 **token_id 数组**。

它来自 `pi0_fast.py` 的自回归生成过程：

```python
token = argmax(last_logit)
output_tokens = put_along_last_axis(output_tokens, ..., token)
```

也就是：

```text
模型 logits
    ↓
argmax / sample
    ↓
token_id
    ↓
output_tokens
```

最后这些 token ids 被解码成字符串，再从字符串里提取 `"Action:"` 后面的 action token。`pi0_fast.py`

---

## 15. FAST action tokenizer 如何 decode 成连续动作？

π0-FAST 不是简单查一个离散 action table。

它流程是：

```text
模型生成 PaliGemma vocab token ids
        ↓
PaliGemma tokenizer decode 成字符串
        ↓
解析 "Action: ... |"
        ↓
把 PaliGemma vocab token ids 映射回 FAST action token ids
        ↓
FAST tokenizer decode
        ↓
continuous actions: [action_horizon, action_dim]
```

关键代码：

```python
return self._fast_tokenizer.decode(
    [action_tokens.tolist()],
    time_horizon=action_horizon,
    action_dim=action_dim
)[0]
```

`tokenizer.py`

所以 FAST tokenizer 的功能是：

```text
encode:
    continuous action chunk → discrete action tokens

decode:
    discrete action tokens → continuous action chunk
```

---

## 16. FAST 和普通 binning action tokenizer 的区别

普通 binning 是：

```text
每个 action 维度独立分桶
token id = 某个 bin
decode = bin_id → 连续值
```

openpi 里 `BinningTokenizer` 就是这样：

```python
action_tokens = action_tokens[: (action_horizon * action_dim)].reshape(
    [action_horizon, action_dim]
)
return action_tokens / self._n_bins * 2 - 1
```

`tokenizer.py`

但 FAST 不是简单 per-dimension binning。它更像一个压缩式 action sequence tokenizer：

```text
整个 action chunk
        ↓
压缩编码成一串 action tokens
        ↓
decode 时整段重建 continuous action trajectory
```

所以 FAST token 不一定一一对应某个时间步、某个 action 维度。

---

## 17. 为什么 action tokenizer 是 vector → id，decode 是 id → vector？

因为这里的 encode/decode 是 **tokenizer / codec 语义**，不是 Transformer 结构语义。

文本 tokenizer：

```text
text → token ids → text
```

动作 tokenizer：

```text
action vectors → token ids → action vectors
```

LLM 擅长预测 token id，所以 π0-FAST 要先把连续动作变成 token id：

```text
continuous action chunk
        ↓ FAST encode
action token ids
        ↓ LLM next-token prediction
predicted action token ids
        ↓ FAST decode
continuous action chunk
```

这和 decoder-only LLM 不矛盾。

---

## 18. 没有 high-level semantic step 标注，怎么训练 policy steps？

如果数据没有 semantic step label，就不能直接对 step text 算 CE。

可选做法有：

```text
1. 有 step label 的样本:
       L = L_action + λ L_step_CE

2. 没有 step label 的样本:
       只算 L_action

3. 用 VLM/LLM/video captioner 生成 pseudo-label:
       轨迹 → 自动标注子任务文本

4. 学 latent plan:
       不输出可读文本，只学内部 planning 表示
```

π0 v1 没有明确做显式 step generation。  
π0.5、LoHoVLA、OneTwoVLA、HiVLA 这类后续模型更明显地走向：

```text
high-level semantic planning
        +
low-level action generation
```

---

## 19. 机器人训练数据与“抓到了没抓到”的问题

π0 这类模型主要是 imitation learning / behavior cloning，不是用 reward 训练。

训练时不一定有：

```text
抓到了: +1
没抓到: 0
```

而是有专家示范：

```text
image_t + state_t + language
        ↓
expert action_t
```

模型学的是：

```text
在这个状态下专家会怎么动
```

评测时才用成功率判断任务完成情况。

在真实机器人上，成功判断可能来自：

```text
人工标注
外部相机
物体位置检测
夹爪状态
力/触觉传感器
任务结果检测
```

如果没有触觉/力反馈，模型对“抓稳没抓稳”的判断是不完整的。  
例如：

```text
夹住了但没夹稳
物体滑落
```

这种情况仿真可以部分模拟，但真实世界仍然会有 sim-to-real gap，尤其受摩擦、材质、接触点、质量分布、夹爪软垫等影响。

---

## 20. Diffusion / Flow 中几种 prediction target

在 diffusion 里常见三种预测目标：

### noise-predict / ε-pred

```text
模型预测加进去的噪声 ε
```

典型代表：

```text
DDPM
Improved DDPM
Guided Diffusion
早期 Stable Diffusion v1
```

### x-predict / x0-pred

```text
模型直接预测干净数据 x0
```

典型代表：

```text
DDIM 采样中显式估计 x0
EDM denoiser
Consistency Models
```

### velocity-predict / v-pred

```text
模型预测 x0 和 ε 的混合速度参数 v
```

典型代表：

```text
Progressive Distillation
Imagen
Stable Diffusion 2.x / SDXL 常见 v-parameterization
```

### flow matching velocity

```text
模型预测路径速度 dx/dt
```

典型代表：

```text
Flow Matching
Rectified Flow
π0 action flow
```

π0 更接近：

```text
flow velocity prediction
```

而不是传统 DDPM 的 ε-pred。

---

# 最终一句话

π0 可以理解为：

> 一个基于 PaliGemma VLM 的 robot foundation policy，用 action expert 和 flow matching 把视觉-语言条件转换成连续动作 chunk；原版 π0 不显式生成 high-level policy steps，openpi 中的 CE loss 主要属于 π0-FAST 路线，用来预测离散 action tokens，而不是预测自然语言 policy steps。

如果只记一个结构，可以记成：

```text
π0:
    image + language + robot state
        ↓
    PaliGemma semantic backbone + action expert
        ↓
    flow matching velocity
        ↓
    continuous action chunk

π0-FAST:
    image + task + state
        ↓
    autoregressive LLM token prediction
        ↓
    FAST action tokens
        ↓
    FAST decoder
        ↓
    continuous action chunk
```



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)