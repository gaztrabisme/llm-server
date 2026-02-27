
r/LocalLLaMA icon
Go to LocalLLaMA
r/LocalLLaMA
•
7d ago
AccomplishedLeg527
How to run Qwen3-Coder-Next 80b parameters model on 8Gb VRAM
Discussion

I am running large llms on my 8Gb laptop 3070ti. I have optimized: LTX-2, Wan2.2, HeartMula, ACE-STEP 1.5.

And now i abble to run 80b parameters model Qwen3-Coder-Next !!!

Instruction here: https://github.com/nalexand/Qwen3-Coder-OPTIMIZED

It is FP8 quant 80Gb in size, it is impossible to fit it on 8Gb VRAM + 32Gb RAM.

So first i tried offloading to disk with device="auto" using accelerate and i got 1 token per 255 second :(.

Than i found that most of large tensors is mlp experts and all other fit in 4.6Gb VRAM so i build custom lazy loading for experts with 2 layers caching VRAM + pinned RAM and got up to 85% cache hit rate and speed up to 1.2t/s it`s 300x speedup.

I wonder what speed will be on 4090 or 5090 desktop..

self.max_gpu_cache = 18  # 
TODO: calculate based on free ram and context window size
self.max_ram_cache = 100 # 
TODO: calculate based on available pinable memory or use unpinned (slow)

Tune this two parameters for your RAM/VRAM (each 18 it is about 3GB). For 5090 max_gpu_cache = 120 and it is >85% cache hit rate. Who can check speed?

Best for loading speed: PCE 5.0 Raid 0 up to 30Gb/s NVME SSD.

Available pinable ram (usualy 1/2 RAM) with DMA - much faster than RAM.

Hope 5090 will give > 20 t/s..
UPD: Added new file modeling_quen3_next_18_02_2026_ssd_only.py

With pinned buffer loading speed from ssd increased 2x. Now 1.07 t/s from ssd without cache (before was 0.49 t/s). With caching total speed increased from 1.2 t/s to 1.53 t/s.

# --- OPTIMIZED CACHE SETUP ---
self.max_gpu_cache = 18   # 1 = ~0.15Gb
self.max_ram_cache = 100  # 1 = ~0.15Gb

# vram + pinned ram [Stats] Tokens: 152 | Time: 99.41s | Speed: 1.53 t/s
self.use_only_ssd = True  # if True - no cache [Stats] Tokens: 256 | Time: 239.72s | Speed: 1.07 t/s
self.use_ram = False  # if False - only VRAM cache [Stats] Tokens: 422 | Time: 290.53s | Speed: 1.45 t/s

    You can run 80b parameters model on 6Gb VRAM with this config, used pinned gpu buffer for fast loading from ssd, used 4.6 Gb VRAM, 0.15Gb pinned RAM buffer, with more free VRAM you can use longer context

119
u/kiloCode avatar kiloCode
•
Promoted
Your favorite AI coding agent, now in Slack
Sign Up
kilo.ai
Thumbnail image: Your favorite AI coding agent, now in Slack
Sort by:
Comments Section
u/AccomplishedLeg527 avatar
AccomplishedLeg527
OP •
7d ago

If I had a 4090 or similar card with FP8 calculations, I would optimize it even faster, since now the cache is in FP8 and calculations in BFP16 it adds overhead for type conversions, but it gives higher accuracy and more expert weights can be placed in the VRAM/RAM cache.
6
Protopia
•
7d ago
• Edited 7d ago

    Any chance that you can upload the processed LLM to HF to save us all having to run the model through the python code?

    How much VRAM + RAM does it use? Is it worth quantising down from FP8 to e.g. int8 in order to reduce the memory still further?

I have a GPU with 6GB of VRAM, and currently 32GB of RAM (though I can afford another 32GB now, with a max total of 128GB later if I use all 32GB SO-DIMMs).
5
Roidberg69
•
7d ago

Always love seeing people work on inference optimisation . Good job and thanks
9
lumos675
•
7d ago

I managed to get 50tps with lmstudio but on q_4_m did not try q8
1
u/stacykade avatar
stacykade
•
7d ago

qwen 2.5 coder is legit. running the 32b quant and it handles most of what i throw at it
1
u/durden111111 avatar
durden111111
•
7d ago

Or just load with "--fit on" with llamacpp.
46
u/AccomplishedLeg527 avatar
AccomplishedLeg527
OP •
7d ago

goal to reach max speed, not just offload random tensors, but keep most used experts weights in vram, only 3 gb vram gives 45% cache hit rate, 20gb > 80%
5
u/tmvr avatar
tmvr
•
7d ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Where did you get the idea that -fit is offloading random tensors? That's nonsense.
16
u/AccomplishedLeg527 avatar
AccomplishedLeg527
OP •
7d ago

in original model for each layer all 512 experts weight in one tensor but usualy used only 100-200 of them, if -fit can split tensor on 512 parts and move unused parts to ram it will be much faster on low ram
1
u/durden111111 avatar
durden111111
•
7d ago

Yes. --fit will load all the tensors and context then load as many blocks as possible on the remaining Vram.

I get 30 tks with Qwen coder next Q8K on my 5090
11
u/Dany0 avatar
Dany0
•
7d ago

I second OP. I have a 5090 too but I cba to test his script, but if you get more than the 40-50tkps I saw other people get with optimised llamacpp configs, then it'd be worth trying
2
u/AcePilot01 avatar
AcePilot01
•
7d ago
• Edited 7d ago

Im on a 4090 with teh Q4 XL, I am getting like 10t's how are you getting so fast I know the 5090 is faster, but I think I should be getting double this at least.

GGML_CUDA_GRAPH_OPT=1 ~/llama.cpp/build/bin/llama-server
-m ~/llama_models/qwen3-coder-next-80b/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \

-c 32768 -fa on \

-ngl 23 \

--threads 26 \

--temp 0 \

--cache-ram 0 \

--cache-type-k q8_0 \

--cache-type-v q8_0 \

--host 172.17.0.1 \

--port 8080

Edit, Ok so actually bringing DOWN my CPU count from 26 to 18, actually gained me about 2 to 3t/s

I am at seeing around 15t/s Same 4090 and 64gb of ram
3
u/ArckToons avatar
ArckToons
•
7d ago

I have an RTX 4090, an i7-13700K, and 64GB of 3200MHz RAM, and I can get 900 pp and 36 tg with the updated llama.cpp using UD-Q4_k_XL. Just use --jinja --n-cpu-moe 28 -t 16 -fa on --no-mmap -np 1 -c 100000 --temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 40 -b 1536 -ub 1536.
7
el95149
•
6d ago

Thank you so much for this command, you just opened my eyes! Those --no-mmap and -ub flags just boosted my PP speed by 100%!
1
lumos675
•
6d ago

Huge Thanks for this no mmap first time i tried it ...man this increased the speed by alot on a 5090... i can see now my gpu is used 100% while processing before it was using only 50 percent
1
u/AccomplishedLeg527 avatar
AccomplishedLeg527
OP •
7d ago

than try my code and compare, all calculations in bfp16
3
politerate
•
6d ago

It's on by default no? I mean until you pass a param which would collide with its logic I guess.
1
u/CodeRabbitAI avatar u/CodeRabbitAI
•
Promoted
Cut Code Review Time and Bugs in Half. Sign up for a free trial!
Sign Up
coderabbit.ai
Thumbnail image: Cut Code Review Time and Bugs in Half. Sign up for a free trial!
IulianHI
•
7d ago

clever approach with the cache tiers. the hit rate makes sense given how MoE routing works - most tokens only hit a few experts anyway. 85% on 3GB is solid tbh
12
u/AccomplishedLeg527 avatar
AccomplishedLeg527
OP •
7d ago

45% on 3GB VRAM + 40% on 15Gb pinned RAM with DMA in total 85%, only 15% constantly reading from disk, and i do not read all espert weights only what needed, usualy most prompts use not more than 65% experts so from 75Gb experts weights only 40-50Gb needed, 20GB on Vram + 30 on pinned ram, perfect for 5090 + 64GB RAM systems
3
u/raysar avatar
raysar
•
6d ago

Why lmstudio and llama.cpp is so basic and don't do your optimization? so many optimization are possible with all moe models.
GGUG offoad to ram on qwen80b is so slow on lmstudio !
3
u/fulgencio_batista avatar
fulgencio_batista
•
7d ago

I get 6tok/s with an RTX3070 and 64GB of DDR4 3600MHz ram with default settings in LM studio. You could get seriously better speeds if you can get more ram instead of offloading to disk.
4
u/AccomplishedLeg527 avatar
AccomplishedLeg527
OP •
7d ago
• Edited 7d ago

i am running on 32 gb ram laptop!! with half PCE bus
Comment Image

and 15% mlp experts reading from disc all time it adds 65% overhead + io waits, also half PCE bus and laptop gpu, x2 if desktop gpu, x2 if PCE 4.0 x16, and x5 if enough ram ~ 10 - 20 t/s should be on your system
4
Borkato
•
7d ago

u/AccomplishedLeg527 avatar
Comment Image
u/Borkato avatar

u/AccomplishedLeg527 avatar

u/IrisColt avatar

u/AccomplishedLeg527 avatar
AccomplishedLeg527
OP •
4d ago

I tested real speed of my 3070ti laptop with this torch lib and bf16 calculations, i loaded only 1 expert per layer just to test max speed (like all fit in vram) and i got only 1.74 t/s. It just slow laptop gpu in bf16 calculations..
1
u/IBKR_Official avatar u/IBKR_Official
•
Promoted
Looking to hedge against inflation or volatility? You can trade physical gold and metals at IBKR with low commissions.
Learn More
interactivebrokers.com
Thumbnail image: Looking to hedge against inflation or volatility? You can trade physical gold and metals at IBKR with low commissions.
u/Longjumping-Elk-7756 avatar
Longjumping-Elk-7756
•
7d ago

I get 28 tokens per second with a Ryzen AI 370, an RTX 3090, and DDR5 RAM. Inference with LM Studio and QWEN 3 Coder Next Q4 KM is completely unusable with OpenCode; it's far too slow.
2
u/tmvr avatar
tmvr
•
7d ago
• Edited 7d ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Did you try the latest llamacpp releases? With b8053, a 4090 and 64GB DDR5-4800 the Q4_K_XL does 48 tok/s max in llama-bench. With context set to 128K it does 42-43 tok/s when actually using it in llama-server.
2
Protopia
•
6d ago

@u/AccomplishedLeg527 I note that qwen3.5 was released today, Andy it is significantly faster.

So if you can work your magic on this so that it can run reasonably well on a 6gb laptop GPU that would be brilliant and it might perform faster as well.

P.s. if you do do your magic on this, can you upload it to HF?
2
u/AccomplishedLeg527 avatar
AccomplishedLeg527
OP •
6d ago

it is 807 GB i have 1 tb ssd and 70 gb free space.. :(
1
Protopia
•
6d ago

Yes. But I imagine that there will be a smaller 80B version very soon.
1
u/Previous_Sail3815 avatar
Previous_Sail3815
•
7d ago

Been running models at similar speeds when pushing VRAM limits and 1.2 tok/s is more usable than it sounds for certain things. I was running a heavily quantized 70b for code review, paste a function, go grab coffee, come back to a solid analysis. latency killed interactive coding but for batch-style tasks where you're not watching each token it worked fine.

Does the 85% cache hit rate hold up with longer contexts though? in my experience with MoE models once you're past like 4k tokens the expert activation patterns get way more diverse and caching gets less effective. were you testing with short prompts or full files?

300x over naive disk offloading is wild. haven't seen the per-layer expert caching approach done like this before.
3
u/AccomplishedLeg527 avatar
AccomplishedLeg527
OP •
7d ago

each layer (48 total) have 512 experts each, cashing works on each layer independently and only holds 18 of 512 in vram that most used. And this covers 45% cache hits using 3Gb vram (18*48 experts in vram). deeper layers activates less experts, some experts can be activated 1600 times when other only once per prompt. Diference between 3Gb and 18Gb for cache 2x cache hit rate, so you can reduce speed 2x but get 18Gb more memory for context. Longer context allways slower but how much slower depends on how optimal cache used.
1
fragment_me
•
7d ago
• Edited 7d ago

I think people are failing to understand that you are keeping part of the model on disk.

EDIT: Have you tried experimenting with llama-cpp or llama-server's -ot parameter? You may be able to accomplish this with that too. E.g. "-ot .ffn_.*_exps.=CPU" I've personally found better performance with using my own offloading with -ot instead of relying on --fit on.
7
u/AccomplishedLeg527 avatar
AccomplishedLeg527
OP •
4d ago

share your run comand +system spec + speed
1
fragment_me
•
3d ago
• Edited 3d ago

Specs: 64GB DDR3, RTX 5090.

The performance of Q4 was very usable. 500-600 pp with 30 tg. Q5-6 prompt processing was just too low though, 200ish PP with 10-20 tg. Note I used back ticks because this is powershell.

.\llama-server.exe -m E:\lm-models\unsloth\Qwen3-Coder-Next-GGUF\Qwen3-Coder-Next-IQ4_XS.gguf  `
 -ot "\.([2-9][0-9])\.ffn_(gate|up|down)_exps.=CPU" `
 --no-mmap --jinja --threads 8  `
 --cache-type-k q8_0 --cache-type-v q8_0  --flash-attn on  --ctx-size 200000 -kvu `
 --temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.01  `
 --host 127.0.0.1   `
 
 
.\llama-server.exe -m E:\lm-models\unsloth\Qwen3-Coder-Next-GGUF\Qwen3-Coder-Next-UD-Q4_K_XL.gguf  `
 -ot "\.([2-9][0-9])\.ffn_(gate|up|down)_exps.=CPU" `
 --no-mmap --jinja --threads 8  `
 --flash-attn on  --ctx-size 65536 -kvu `
 --temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.01  `
 --host 127.0.0.1`
 
.\llama-server.exe -m E:\lm-models\unsloth\Qwen3-Coder-Next-GGUF\Qwen3-Coder-Next-UD-Q4_K_XL.gguf  `
 -ot "\.([2-9][0-9])\.ffn_(gate|up|down)_exps.=CPU" `
 --no-mmap --jinja --threads 8  `
 --cache-type-k q8_0 --cache-type-v q8_0  --flash-attn on  --ctx-size 200000 -kvu `
 --temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.01  `
 --host 127.0.0.1 `
 

.\llama-server.exe -m E:\lm-models\unsloth\Qwen3-Coder-Next-GGUF\Qwen3-Coder-Next-UD-Q5_K_XL.gguf  `
 -ot "\.(19|[2-9][0-9])\.ffn_(gate|up|down)_exps.=CPU" `
 --no-mmap --jinja --threads 8 `
 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on  --ctx-size 200000 -kvu `
 --temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.01  `
 --host 127.0.0.1   `
 
.\llama-server.exe -m E:\lm-models\unsloth\Qwen3-Coder-Next-GGUF\Qwen3-Coder-Next-UD-Q6_K_XL.gguf  `
 -ot "\.(1[8-9]|[2-9][0-9])\.ffn_(gate|up|down)_exps.=CPU" `
 --no-mmap --jinja --threads 8  `
 --cache-type-v q8_0  --flash-attn on  --ctx-size 200000 -kvu `
 --temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.01  `
 --host 127.0.0.1 `

 
.\llama-server.exe -m E:\lm-models\unsloth\Qwen3-Coder-Next-GGUF\Qwen3-Coder-Next-UD-Q6_K_XL.gguf  `
 -ot "\.(1[8-9]|[2-9][0-9])\.ffn_(gate|up|down)_exps.=CPU" `
 --no-mmap --jinja --threads 8  `
  --flash-attn on  --ctx-size 200000 -kvu `
 --temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.01  `
 --host 127.0.0.1  `

2
u/AccomplishedLeg527 avatar
AccomplishedLeg527
OP •
3d ago

i can`t run with "-ot" Q4_K_M model on 8Gb+32Gb not enough memory even with 1024 context, only --fit works, 46gb on 40gb total memory but it works ~10 t/s
1
jacek2023
•
7d ago
emoji:Discord:
Profile Badge for the Achievement Top 1% Poster Top 1% Poster

Llms are large by definition ;)
0
PhotographerUSA
•
7d ago

Why, are you reposting this? Plus it's on Github not hugging face. I wouldn't trust the module probably malicious.
-3
u/AccomplishedLeg527 avatar
AccomplishedLeg527
OP •
7d ago

I am not reposting, i am owner of this repo.
4
u/waiting_for_zban avatar
waiting_for_zban
•
7d ago
emoji:Discord:

So if I understand this correctly, you modified the original qwen3-coder script to modify the way the transformer layers are loaded into vram by ignoring the the mlp expert layers and shoving them to ssd, and prioritizing the attention layers to gpu instead?

This is rather very interesting. I think you need a tad better readme, but you have done quite an awesome job. I might look into this later! I think the rest of the users did not quite understand what you have done, but great job!
9
u/AccomplishedLeg527 avatar
AccomplishedLeg527
OP •
7d ago

Yes all tensors except mlx experts succesfuly fit in 4.6Gb vram, you can run it on 6 Gb. Vram cache and pinable ram cache customizable to any size. The more you reserve memory for cache the more speed you have.
4
u/Several-Tax31 avatar
Several-Tax31
•
6d ago

Running models from SSD is the dream here. 1.2t/s is not that bad, could be useful in some cases. Now apply your magic to Qwen 3.5 😅 
1
PowerSage
•
6d ago

Yeah I second this, this is solid.
2
u/ZealousidealShoe7998 avatar
ZealousidealShoe7998
•
7d ago

would this work on a older gpu like 1080 ti or does it need a a newer gpu architecture
1
u/Creepy-Bell-4527 avatar
Creepy-Bell-4527
•
7d ago

Why?
1
u/R_Duncan avatar
R_Duncan
•
6d ago

It's not the 8GB VRAM (which is however too less to run fp8 @ 128k context, try Q4 models) is the 32 GB cpu ram which is really the bottleneck here. even with mmap feature the Q4 version is slow with 32 GB and just decent with 64Gb. I think 96 or 128GB cpu RAM needed for fp8
1
Shoddy_Bed3240
•
6d ago

Tested Qwen3-Coder-Next q8 on an RTX 5090 (rest loaded in DDR5) — getting ~45 t/s without any optimizations.

One thing I think people misunderstand when talking about CPU limitations for LLM performance is that the discussion usually focuses only on RAM speed. But the CPU itself matters a lot more than many assume.

In practice, a 3–4 year old processor often can’t fully utilize your system’s maximum memory bandwidth, which becomes a real bottleneck. Even if you have fast DDR5, you won’t see the full benefit if the CPU can’t keep up.
1
u/DefNattyBoii avatar
DefNattyBoii
•
6d ago

Did someone test this with 12/24 gb vram gpus, and compared it to Q4 --fit with llama.cpp? Benchmarks would be nice, since there is a minimal but measurable drop with Q4.
1
