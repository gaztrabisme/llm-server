# We got a fan
User Avatar
WarthogConfident4039
2:55 PM
Yo man. I appreciate on your extensive testing of the Qwen 3.5 27B and 35B models. That was glorious.
Could you do another round for the Qwen 3.6 this time around?
User Avatar
gaztrab
3:58 PM
Oh man thanks a lot for the message. That's really dope.
I gotta be honest back then when I did those experiments I had too many coffees lol
But yeah I do plan to do another round comparing 27b Q3 vs 35b Q8 with MTP
Gonna do it this weekend. Stay tuned!
User Avatar
WarthogConfident4039
4:00 PM
Wow thanks. Could you send me a message when it's done with a link?
Thanks a ton.
User Avatar
gaztrab
4:03 PM
Definitely
Now I cannot procrastinate this any longer since I made a promise to you LMAO
User Avatar
WarthogConfident4039
4:04 PM
I love you man.
User Avatar
gaztrab
4:04 PM
I love you too #nohomo

# New development

Go to LocalLLaMA
r/LocalLLaMA
•
3d ago
janvitos
emoji:Discord:
80 tok/sec and 128K context on 12GB VRAM with Qwen3.6 35B A3B and llama.cpp MTP
Tutorial | Guide

Just wanted to share my config in hopes of helping other 12GB GPU owners achieve what I see as very respectable token generation speeds with modest VRAM. Using the latest llama.cpp build + MTP PR, I got over 80 tok/sec with 80%+ draft acceptance rate on the benchmark found here: https://gist.githubusercontent.com/am17an/228edfb84ed082aa88e3865d6fa27090/raw/7a2cee40ee1e2ca5365f4cef93632193d7ad852a/mtp-bench.py

Here's my PC specs:

OS: CachyOS (HIGHLY recommended)
CPU: AMD Ryzen 7 9700X
RAM: 48GB DDR5-6000 EXPO I
GPU: RTX 4070 Super 12GB

Results with other hardware may vary.

To run llama.cpp with MTP support, you need to build it from source and add a draft PR that hasn't yet been merged with the master branch. You can find a very nice guide on how to do that here and also download the Qwen3.6 MTP GGUF: https://huggingface.co/havenoammo/Qwen3.6-35B-A3B-MTP-GGUF - Thanks u/havenoammo!

llama.cpp command:

llama-server \
  -m Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf \
  -fitt 1536 \
  -c 131072 \
  -n 32768 \
  -fa on \
  -np 1 \
  -ctk q8_0 \
  -ctv q8_0 \
  -ctkd q8_0 \
  -ctvd q8_0 \
  -ctxcp 64 \
  --no-mmap \
  --mlock \
  --no-warmup \
  --spec-type mtp \
  --spec-draft-n-max 2 \
  --chat-template-kwargs '{"preserve_thinking": true}' \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0

The most important parameter here is -fitt 1536. Since part of the model is offloaded to CPU because of its size and , this tells llama.cpp to properly balance the load on the GPU/CPU to get the best possible performance, and leaves 1536 MB of free memory for the MTP draft model and KV cache. Since I'm running my dGPU as a secondary GPU (monitor plugged in the iGPU), I can use all the available 12GB VRAM for inference. 1536 might be too small if you use your dGPU as your primary GPU, so test it out first.

You can also try different values for -spec-draft-n-max. I got slightly better tok/sec with 3, but a much better acceptance rate with 2, so the trade off was not worth it. With MTP, you want to maximize speed AND acceptance, so you need to find the best balance between both.

Benchmark results:

mtp-bench.py

code_python        pred= 192 draft= 132 acc= 125 rate=0.947 tok/s=80.8
code_cpp           pred=  58 draft=  40 acc=  37 rate=0.925 tok/s=81.8
explain_concept    pred= 192 draft= 152 acc= 114 rate=0.750 tok/s=70.0
summarize          pred=  53 draft=  40 acc=  32 rate=0.800 tok/s=75.4
qa_factual         pred= 192 draft= 144 acc= 119 rate=0.826 tok/s=77.8
translation        pred=  22 draft=  16 acc=  13 rate=0.812 tok/s=81.9
creative_short     pred= 192 draft= 160 acc= 111 rate=0.694 tok/s=69.2
stepwise_math      pred= 192 draft= 144 acc= 119 rate=0.826 tok/s=76.5
long_code_review   pred= 192 draft= 148 acc= 117 rate=0.790 tok/s=73.2

If you have any questions, feel free to ask :)

Cheers.
635
u/Meshyai avatar Meshyai
•
Promoted
🔥 Up to 70% OFF all plans. 500 bonus credits on Pro Monthly. Biggest sale in Meshy history. Claim now.

    https://www.meshy.ai/anniversary-2026
    https://www.meshy.ai/anniversary-2026
    https://www.meshy.ai/anniversary-2026

meshy.ai
Sign Up
Sort by:
Comments Section
u/StupidScaredSquirrel avatar
StupidScaredSquirrel
•
3d ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Why -no-mmap?
24
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

It's a general llama.cpp recommendation when using --mlock (prevents swapping to disk). --no-map loads the entire model into RAM instead of loading parts when needed. As I understand it, it prevents disk I/O and makes memory usage more predictable. It might result in slower loading times, but better stability during inference.
40
sh4rk1z
•
3d ago
• Edited 3d ago

benchmarked mmap and no-mmap less than an hour ago on rtx2070S/ryzen3950x/64ram with vram limited to 6.25GiB for qwen-3.5-9B-ud-q4-k-xl so I can use my desktop while using the local model. Result over 3 runs with no-mmap gave improvements:

- ~1.5% decode speed improvement.
- ~5.2% prompt processing improvement.
- 28 MB vram less used.
- standard deviation between runs droped by 10-20x
- no disk io so less wear/tear

I'm still experimenting with some things (turboquant, trellis) and will post once done and then try the Qwen 3.6.
27
[deleted]
•
3d ago

u/janvitos avatar

letsgoiowa
•
3d ago

Std?
1
CircularSeasoning
•
3d ago

<think>

The user has entered three letters, "Std", with a question mark, possibly hoping to elicit more information about (Something To Do?) with "Std"? I'm not sure what that means. I should ask for clarification.

Wait! The user's name is 'letsgoiowa' (i.e., "Let's go, Iowa!") so let me research what happens in Iowa in connection with the letters or acronym, "STD"...

[web search content omitted]

Ah.

I should helpfully advise the user to test for: Chlamydia.

All good.

Proceed.

</think>
52
sh4rk1z
•
3d ago

😂😂😂
6
CircularSeasoning
•
3d ago

letsgoiowa looking at me all σ_σ
4
letsgoiowa
•
2d ago

Standard deviation lol

But thanks
4
u/theowlinspace avatar
theowlinspace
•
3d ago

—mmap with —mlock shouldn’t use disk io after you’ve loaded the model because it locks the mapped pages in RAM
1
u/BitGreen1270 avatar
BitGreen1270
•
2d ago

I have a 780m igpu and adding --no-mmap makes it use 2GB extra RAM with nothing else changing. My prompt is just a 500 word story in the style of Roald Dahl. Since I only have 32GB, that's a pickle. No difference in tps though - still getting exactly the same. This is for non-MTP though. I'm downloading the MTP version to try out with your params (thanks so much!)
1
u/StupidScaredSquirrel avatar
StupidScaredSquirrel
•
3d ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

But when is it useful to have mmap then? If having -no-mmap still loads what is needed?
1
farkinga
•
3d ago

When the model is big, and when the weights will be in system ram anyway (e.g. a moe) , use mmap (on Linux) to avoid loading the whole model into ram. With mmap, Linux will load the weights into ram as needed. However, use no-mmap if you have a performance reason to keep the weights in ram anyway. It should run a little faster with no-mmap but it takes longer to start.
9
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

To be honest, I would say try both and see what works best for ya 😄
4
u/Marksta avatar
Marksta
•
3d ago

Because mmap is faster to load the model on Linux where it has a real system level mmap. And if you were to turn off the server and turn it back on again, the model would already be in mmap. Restarting something big like Deepseek without mmap would mean waiting a few minutes each time to load it, unload it, load it again...
1
dark-light92
•
3d ago
llama.cpp

So it doesn't mmap.
-1
zulutune
•
3d ago

Hey OP thank you so much for this. I have an underutilized 5070ti and I’m going to try this out. Hopefully this weekend.
47
zulutune
•
3d ago

Btw did you try DeepSeekV4? I’m kinda curious for this model too.
8
u/janvitos avatar
janvitos
OP •
3d ago
• Edited 3d ago
emoji:Discord:

I've tried DeepSeek V4 cloud for coding and didn't like it at all. It was overthinkig way too much and seemed confused and paranoid. But that's me. I'm sure others would debate this 😄

When using cloud models for coding, GPT 5.5 is my top choice. In my opinion, its deterministic behavior makes it extremely apt at one shotting large and complex code additions/modifications.

To be honest, I found Qwen3.6 35B A3B local to be in the same league as most other and bigger open LLMs, except GLM 5.1, which can debug and resolve issues that Qwen3.6 cannot.
24
u/rz2000 avatar
rz2000
•
3d ago
• Edited 3d ago

Have you tried DeepSeek v4 with different thinking parameters? Using the flash version locally, I’ve found that completely turning off thinking gets good results.

I’ve only used it with chat. In Kagi Assistant which uses fireworks.ai, together.ai, or deepinfra, it can be extremely slow with either the pro or flash version. However the quality of the written analysis is very good with ot without websearch enabled.

Locally, I have used https://github.com/antirez/ds4 to run the flash version. This custom engine achieves pretty excellent performance, and here is where I have found a lot of benefit to simply switching of the reasoning step with \nothink.

I can’t run the full pro version, but it is pretty amazing to get better performance from the flash version than I can get from cloud providers, albeit with Kagi in between.
4
zulutune
•
3d ago

Does that mean you have a macbook with 128GB?
2
u/rz2000 avatar
rz2000
•
3d ago

A Mac Studio with 256 GB. I think Mac Studios with 192GB+, or the maxed out M5 MacBook Pro is what antirez was targeting with this inference engine.

In a couple years this sort of performance will likely be cheap, and it would worry me more if I were Google, OpenAI, Anthropic than some of the other open model releases that suddenly made AI briefly crash.

I haven’t gotten Gemma 4 with MTP acceleration to work very reliably yet, but that is another way that local inference is becoming viable for much more than just hobbyist use.
2
zulutune
•
3d ago

Gratz, you’re in a different league.

So how does Qwen 3.6 and DS4 compare, what’s your favorite? Do you ever feel the need to use cloud models, or does that level of GB’s really give you the raw power of a Opus/Codex?
1
u/rz2000 avatar
rz2000
•
2d ago

I haven’t used either for much code assistance.

The “personality” of DeepSeek v4 is much more like GLM 4.6 or 4.7, which I think is pretty good, but without the need to quantiize it down to 4bits which can result is strange errors. DeepSeek v4 flash fits in 160GB of memory at full precision.

For tasks other than coding I find Qwen pretty unbearable. It seems very incurious and very worried about anything that might be innovative.
2
zulutune
•
2d ago

Interesting observation haha :) thanks for sharing your insights
1
zulutune
•
3d ago

Thanks for posting this!
2
Still-Notice8155
•
3d ago
• Edited 3d ago

Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf on GTX 1070 8GB + i7-11700 16GB

Config: turboquant+MTP | n-cpu-moe 32 | turbo4/turbo3 KV | ctx 131K | ctx-checkpoints 8

---

Gen t/s degradation (attention O(n) cost):

0K: 48 t/s ████████████████████████████████

10K: 31 t/s █████████████████████

30K: 28 t/s ██████████████████

50K: 23 t/s ███████████████

80K: 23 t/s ███████████████ ← DeltaNet plateau

100K: 19 t/s ████████████

125K: 13.6 t/s █████████

Curve flattens 30-80K thanks to 30 DeltaNet O(1) layers. Only 10 attention layers drive degradation.

PP t/s (batch-driven, unaffected by context):

Short prompt (<20 tok): 41 t/s avg — overhead bound

Batched prompt (50+ tok): 135 t/s avg — GPU parallel

At 125K ctx: still 78-95 t/s PP

Draft acceptance: 58-86% depending on task predictability. Lifetime: ~90%.

VRAM: 7.5 GB used, 633 MB free at 131K. Turbo4/turbo3 KV = 590 MB (vs 720 MB q4_0).

RAM: 12 GB used (model no-mmap = 13.2 GB + MoE CPU offload + 500 MB prompt cache). 2 GB free with checkpoints=8.

Improvement over non-MTP baseline:

Non-MTP MTP+turbo Speedup

5K: 27.4 → 48 = 1.8x

80K: ~7 → 23 = 3.3x

125K: ~3 → 13.6 = 4.5x

The gap widens at high context — MTP saves ~constant time per token regardless of context, while attention cost grows linearly.
17
DunderSunder
•
3d ago

    Qwen3.6-35B-A3B-MTP

which quant is this?
5
Still-Notice8155
•
3d ago

Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf. I would love to test the Q4_K_M, but I don't have enough RAM for now.

--n-cpu-moe 32 --no-mmap --parallel 1 \

--ctx-checkpoints 8 \

--spec-type mtp --spec-draft-n-max 3 \

--cache-type-k turbo4 --cache-type-v turbo3 \

--jinja -c 131072 -fit off
3
Still-Notice8155
•
2d ago

Qwen3.6-35B-A3B IQ4_XS + MTP on GTX 1070 8GB

Hardware

CPU: i7-11700 (8c/16t)

RAM: 32 GB DDR4-3200

GPU: GTX 1070 8GB (Pascal, stock clocks, no OC)

OS: Ubuntu 26.04, CUDA 12.4, driver 580.142

Model

Name: Qwen3.6-35B-A3B (MoE, 256 experts, 8 active, 3B active params)

Quant: IQ4_XS (19.4 GB, 4.37 BPW)

MTP: Q8_0 draft heads, 3-token speculative decoding

Arch: 30 DeltaNet (O(1)) + 10 quadratic attention (O(n)) layers

Context: 131,072 tokens

Server Flags

--n-cpu-moe 35 --no-mmap --parallel 1 --ctx-checkpoints 32

--spec-type mtp --spec-draft-n-max 3

--cache-type-k turbo4 --cache-type-v turbo4

--jinja -c 131072 -fit off

Build: llama.cpp master + PR #22673 (MTP) + turboquant cache patches

Turbo4 KV cache: 4-bit WHT quantization for K and V

Gen Speed vs Context

0–15K: 32.1 t/s

15–40K: 28.1 t/s

40–70K: 24.3 t/s

70–100K: 23.0 t/s

100–131K: 18.1 t/s

Prompt Processing

0–15K: 148 t/s

40–70K: 107 t/s

100–131K: 64 t/s

Draft Acceptance (MTP)

Per-task: 42–89% (varies by difficulty)

Global: 75–80% lifetime

VRAM at 131K

GPU model: 4,578 MB

KV cache: 1,122 MB (turbo4 compressed)

Recurrent: 251 MB

Compute: ~493 MB

Total: ~7.6 GB / 475 MB free

RAM

22 GB used / 9 GB free (32 GB total, --no-mmap)

Have retested in 32GB ram. Still good performance. I'm not sure about the quality degredation.
1
Still-Notice8155
•
2d ago

I have tried this benchmark https://github.com/alexziskind1/codeneedle

## Qwen3.6-35B-A3B IQ4_XS + MTP — CodeNeedle Positional Recall

Tests exact line-by-line recall: stuff entire source into context, reproduce

functions verbatim. Pass = ≥8/20 lines match exactly including whitespace.

MTP speculative decoding at n=3, turbo4 quantized KV cache.

### Results

HTTP no-think: 10/11 PASS (91%), 187/220 lines (85%), 50 total hallucinations

HTTP think: 9/11 PASS (82%), 186/220 lines (85%), 66 total hallucinations

jQuery no-think: 14/16 PASS (88%), 283/320 lines (88%), 319 total hallucinations

jQuery think: 14/16 PASS (88%), 271/320 lines (84%), 43 total hallucinations

### MTP Draft Acceptance

Global Per-task range

HTTP no-think 94% 86-100%

HTTP think 93% 86-100%

jQuery no-think 91% 51-100%

jQuery think 87% 62-100%
1
u/FirefoxMetzger avatar
FirefoxMetzger
•
3d ago

what does the turboquant refer to here? K/V cache or or model quantization?
2
Still-Notice8155
•
2d ago

KV cache.
4
FrostWolfDota
•
3d ago

I have a 16GB AMD cpu, will try to reproduce it when I find some time. Never tried using llama.cop directly, only through LM studio.
13
u/house_monkey avatar
house_monkey
•
3d ago

Wish I could reproduce my 16GB AMD cpu
14
429_TooManyRequests
•
3d ago

Wow this post is perfect timing. I have a 3080 Ti and was depressed I couldn’t get this exact model working last night. I’ll try it out today and send results!
5
Independent-Flow3408
•
3d ago

This is a really useful writeup, thanks.

The "-fitt 1664" detail is the part I would have missed. For long-context coding workflows, did you notice the speed dropping mainly from KV/cache pressure, or from CPU/GPU balancing once the context gets large?

Also curious if you tested this with an agent workflow like OpenCode/Continue, or only direct llama.cpp prompting.
5
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

Speed drops towards 50 tok/sec when context has filled up near 128K. But that's still very reasonable and usable. Didn't notice quality degradation.

I've been using this with Opencode for the past few days without any issues. I can analyze the entire codebase of a small project, which fills up the context near 75K, and continue working on it normally no problem.

So yeah, I would consider this as pretty stable 😄
4
ai_without_borders
•
3d ago

the 80 tok/s is with 128K context loaded — at shorter contexts (4-8K) you would be pushing 100+ easily. MTP overhead shows up more in prompt processing than in token generation, so the win is biggest on long generation runs vs short QA bursts. good config though, -no-mmap with mlock is the right call for sustained throughput.
5
u/auriko_ai avatar u/auriko_ai
•
Promoted
The cheapest LLM provider might not be the cheapest for you.
ElChupaNebrey
•
3d ago

What is you speed on 27b
4
twiddlebit
•
3d ago

27b wont fit on 12gb of vram so probably not very good
9
u/janvitos avatar
janvitos
OP •
3d ago
• Edited 3d ago
emoji:Discord:

I haven't even tried it after seeing other people's benchmarks. I know it wouldn't be fast enough for real-world coding anyways, so I'll wait until some miracle happens or I buy a new GPU 😄
5
u/ducksoup_18 avatar
ducksoup_18
•
3d ago

I have 2 3060s for a total of 24gb vram. I'd love to see these kind of numbers with that setup. Will try.
4
HavenTerminal_com
•
3d ago

the spec-draft-n-max 2 vs 3 finding is the kind of thing you only figure out by running both. appreciate you logging it.
4
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

And I recommend that everyone test their own values, as I've seen others find success with 3 or 4 😄
1
u/MistingFidgets avatar
MistingFidgets
•
2d ago

Spec Decode and MTP are really awesome. I have some benchmark data i want to share but can't post yet, need some upvotes on comments before localllama will let me.... help me out here
4
u/Shaped_ai avatar u/Shaped_ai
•
Promoted
Stop drowning your local weights in 50k tokens of RAG noise.
Fuzilumpkinz
Cake icon •
3d ago

I’ll try this for sure. I’m getting 40 atm but I’m on a 6700 xt. Curious if I can find any increases
3
slimdizzy
•
3d ago

I have a 3080 12gb I will try this on. Thanks muchly OP!
3
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

Awesome! Share your results 😄
3
burdzi
•
3d ago

Nice 🤩 does MTP also work for vision? If I give it images?
3
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

There seems to be some issues with vision at the moment. You can read about it here: https://github.com/ggml-org/llama.cpp/issues/22867 and on the official PR thread: https://github.com/ggml-org/llama.cpp/pull/22673
4
masterlafontaine
•
3d ago

What is the prompt speed? Usually this is what makes agentic code the most boring and slow. It's usually about reading, say 50k, then writting 3k.
3
sirnixalot94
•
3d ago

I haven’t tried MTP yet, but I have that same model running on an RTX 4080 16GB with —cpu-moe=20 (Ryzen 9 5950X and 64GB system RAM) and I’m getting 105t/s pp and right at 50t/s generation speed. I’m going to check this out and see if adding this in addition to that will improve my performance even more. Thanks for the findings!
3
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

Definitely try the -fitt flag. It replaces --cpu-moe and the guessing work. The only thing you need to figure out is the right amount of reserved RAM. So for non-MTP, I started with -fitt 256, but ran into OOM errors here and there. It was rock solid with -fitt 512. You can check your VRAM usage with nvidia-smi. For me, 11800MiB / 12282MiB is pretty much the max I can push.
3
cognitium
•
3d ago

Are you actually getting good output from that model though? It's the fastest local model I've ever used because only 3B are active at a time but it'll use half of it's context endlessly soliloquizing about how it's a good model that follows the rules and then doesn't follow them.
3
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

Try it with thinking disabled: --chat-template-kwargs '{"enable_thinking": false}' (might be slightly different for Windows).

I feel it's pretty darn good at coding this way. Make sure you use the right launch parms for instruct / non-thinking mode though: temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
3
cognitium
•
3d ago

Alright, I'll try those. I spent most of yesterday playing with qwen3.6 35B and 27B and they both have issues with over thinking. The speed of 35B is what's most impressive.
1
Substantial-Thing303
•
2d ago

you can also try deepseek's scratchpad grammar on qwen3.6 to cut down on the thinking: https://github.com/noonghunna/club-3090/blob/master/docs/STRUCTURED_COT.md
1
_bones__
•
2d ago

Getting 60t/s on an RTX3080 12GB with this setup. So quite useful!

I am getting a huge preprocessing time in an existing session, which is a bit weird, as I didn't have that with regular Qwen 3.6 before this, a Q3 that got me 45t/s.

Definitely interesting stuff, thanks for posting.
3
u/alchninja avatar
alchninja
•
3d ago

Hey, thanks for the info! Could I ask what your CPU and RAM specs are? I'm on a Ryzen 5700x and 32GB DRR4-3600, just trying to get a feel for how much people are able too benefit from having newer CPUs and DDR5.
2
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

Here's my specs:

AMD Ryzen 7 9700X
48GB DDR5-6000

I'm surprised I'm not encountering more issues with the 3 x DIMM RAM config. It's actually running great even with EXPO I 😄

I was able to run the same model (non-MTP) with 32GB, but it was tight. That's why I stole a 16GB DIMM from my son's gaming PC. With 48GB, I have a 10-12 GB buffer at all times when the model is loaded.

One thing to note, since installing CachyOS, I noticed it's way less RAM hungry than Windows. And to be honest, once everything is setup properly, CachyOS is pretty incredible. It's actually my daily driver now. I haven't switched back to Windows in days.
3
u/alchninja avatar
alchninja
•
3d ago

Thanks! I bet your son is super happy about his missing RAM stick lol

Yep, getting into local LLMs and seeing how Kubuntu breathed new life into my 9 year old Dell XPS (I don't know how I lived without KDE Plasma for so long) finally pushed me away from Windows for good on all my machines. I still keep it on a partition just for the occasional gaming session with a friend (unfortunately the stuff we play needs Windows) but I can't imagine ever using it as my daily again.
2
Sufficient_Sir_5414
•
3d ago

How are you balancing the KV cache for the 128k context window alongside the MTP draft model on only 12GB? Did you have to aggressively tune the -fitt parameter or sacrifice context depth to maintain that 80% acceptance rate?
2
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

That's the magic of -fitt: Once you find the sweet spot that doesn't cause any OOM, you get a rock solid local inference setup that can perform very well even on a hybrid GPU/CPU config. No long tuning. No sacrifices. Just a few code analysis / creation runs with the agent to fill the context and test the VRAM limit.
1
u/coolaznkenny avatar
coolaznkenny
•
3d ago

Going to utilize this guide once i get my hands on a steam machine!
2
u/FirefoxMetzger avatar
FirefoxMetzger
•
3d ago

Hm, so the reason this works as well as it does is that you offload layers to host memory (i.e. your total footprint is >12GB) and you increase decode tok/s with speculative decoding using a draft model?
2
u/janvitos avatar
janvitos
OP •
2d ago
emoji:Discord:

Exactly!
2
oviteodor
•
2d ago

Thank you OP
2
u/BitGreen1270 avatar
BitGreen1270
•
2d ago

This is very cool, thanks for sharing. I used the same prompt on the non-MTP and the MTP version and got the following:

Non-MTP - [ Prompt: 80.3 t/s | Generation: 21.6 t/s ]

MTP - [ Prompt: 71.9 t/s | Generation: 28.1 t/s ]

Prompt speed seems to have gone down, but token generation has gone up significantly. This is on my 780m iGPU.
2
u/pwmcintyre avatar
pwmcintyre
•
2d ago

legend! i'm finally getting useful results on my 4070 12GB
2
u/chille9 avatar
chille9
•
2d ago

50 t/s with rtx 4060Ti 16Gb and 32gb ram! Also using the q5 quant at a 98k context! Magnificent.
2
Loouiz
•
17h ago

Is it stable? Did you make any other adjustments? I'm trying this with a 16gb 4080 super an 32gb ram and I'm gettin oom here and there...
1
u/chille9 avatar
chille9
•
8h ago
• Edited 8h ago

I´ve made very small adjustments. I also recompiled using the instructions that op had listed.

Here´s the bat file i run in my llama dir where you can see my settings.
https://pastebin.com/dSkkKX60

It´s been pretty stable for me. I hope you can solve it! Only getting oom or errors on using text files and pdfs. Pdfs and text works great on the Q4 qwen 35B MTP model.

Edit: put --spec-draft-n-max to 3 instead of 2 and no crashes with pdfs.
1
b0ts
•
2d ago

On my 3070 (8GB) with a Ryzen 9 7900x and 64GB DDR5 6400:
Comment Image
2
u/RaspNAS avatar
RaspNAS
•
1d ago
• Edited 1d ago

I tried the MTP benchmark on llama.cpp too after seeing your post.
Thanks a lot! This ultra-high-speed LLM is insane !!!!
Hardware:

    GPU: RTX 3060 12GB

    CPU: Ryzen 9 5950X (16 threads)

    RAM: DDR4-3200 40GB

    OS: Windows 11 Pro (on Proxmox with PCIe Passthrough)

Administrator in 🌐 letwir-main in ~\Documents via  v24.14.0 via 🐍 v3.14.2 (.venv)
❯ curl https://gist.githubusercontent.com/am17an/228edfb84ed082aa88e3865d6fa27090/raw/7a2cee40ee1e2ca5365f4cef93632193d7ad852a/mtp-bench.py -o mtp-bench.py
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  7709  100  7709    0     0  77194      0 --:--:-- --:--:-- --:--:-- 79474

Administrator in 🌐 letwir-main in ~\Documents via  v24.14.0 via 🐍 v3.14.2 (.venv)
❯ sd "8080" "11434" .\mtp-bench.py

Administrator in 🌐 letwir-main in ~\Documents via  v24.14.0 via 🐍 v3.14.2 (.venv)
❯ py .\mtp-bench.py
  code_python        pred= 192 draft= 156 acc= 138 rate=0.885 tok/s=38.9
  code_cpp           pred= 192 draft= 180 acc= 131 rate=0.728 tok/s=35.0
  explain_concept    pred= 192 draft= 189 acc= 128 rate=0.677 tok/s=33.7
  summarize          pred=  53 draft=  48 acc=  36 rate=0.750 tok/s=37.4
  qa_factual         pred= 192 draft= 180 acc= 131 rate=0.728 tok/s=35.2
  translation        pred=  22 draft=  24 acc=  13 rate=0.542 tok/s=31.6
  creative_short     pred= 192 draft= 207 acc= 122 rate=0.589 tok/s=31.1
  stepwise_math      pred= 192 draft= 174 acc= 133 rate=0.764 tok/s=35.8
  long_code_review   pred= 192 draft= 192 acc= 127 rate=0.661 tok/s=32.8

Aggregate: {
  "n_requests": 9,
  "total_predicted": 1419,
  "total_draft": 1350,
  "total_draft_accepted": 959,
  "aggregate_accept_rate": 0.7104,
  "wall_s_total": 46.07
}

build options:

.\vcpkg install  pthreads openssl curl[core,http2,http3,openssl,ssh,zstd] --triplet x64-windows
git fetch origin pull/22673/head:mtp-clean
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_F16=ON -DGGML_CUDA_GRAPHS=ON -DCMAKE_TOOLCHAIN_FILE="C:/PATH/vcpkg/scripts/buildsystems/vcpkg.cmake"

add options: --threads 16 --threads-batch 16
change options: --spec-draft-n-max 3

llama-server --port 11434 --host 0.0.0.0 --threads 16 --threads-batch 16 -m "A:\LLM\Qwen3.6-35B-A3B-MTP-UD-Q3_K_XL.gguf" -fitt 1736 -c 131072 -n 32768 -fa on -np 1 -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 -ctxcp 64 --no-mmap --mlock --no-warmup --spec-type mtp --spec-draft-n-max 3 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --jinja --webui-mcp-proxy

2
u/eliko613 avatar
eliko613
•
1d ago

Great writeup — the -fitt tuning is genuinely underappreciated. Most people just set -ngl 99 and wonder why their CPU is saturated.

A few things that helped me squeeze out a bit more on a similar split setup:

    Bumping --ctxcp slightly (128 worked better for me than 64 at longer context) — worth benchmarking your specific use case

    --spec-draft-n-max 2 is conservative; if your draft model is fast you can push to 3–4 and get meaningful throughput gains

    With preserve_thinking: true the KV cache fills up fast at 131k context — make sure you're actually using that window or trim -c to free headroom

Also been using zenllm.io for quick parameter testing before committing to long runs — handy for dialing in temp/top-p without burning local resources. Not affiliated, just a useful scratch pad.

What's your tok/s looking like on this config?
2
u/zerozero023 avatar
zerozero023
•
22h ago

Nice write-up. The -fitt flag is something I never paid attention to before — makes sense for hybrid GPU/CPU setups. Did you notice any quality difference with Q4_K_XL vs higher quants at this context size?​​​​​​​​​​​​​​​​
2
u/q-admin007 avatar
q-admin007
•
19h ago

Awesome work. I have a 5070 Ti 16GB connected via Oculink with a Strix Halo. Will give it a go later with UD-Q6_K_XL. It seems to be the sweetspot in terms of precision on smaller systems. I also would rather half my context and use f16 there.
2
admajic
•
3d ago

Huh? On a 3090 I'm getting average 150 tok/s and tops at 200 tok/s. Amazing how offloading destiny's u
4
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

That's awesome! It's actually because the entire model fits into your VRAM, which is impossible on a 12GB GPU.
12
u/PrometheusZer0 avatar
PrometheusZer0
•
3d ago

what's your setup? Lucebox?
1
admajic
•
2d ago

Using mtp you need to pull it from git i did a write-up about it
2
yoomiii
•
2d ago

wake me up when MTP PR is merged
2
u/damianzoys avatar
damianzoys
•
3d ago
• Edited 3d ago

I got some nice tok/s too, but the hallucinations make it almost impossible to use. It hallucinates tools and directories which aren’t there, even with low temperature. Any idea how to fix this?
1
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

Are you getting these hallucinations with MTP only?

To be honest, I haven't noticed any issues with MTP and have been using it for a few days to do some code work, but no major project yet. No tool issues at all. For my setup, Qwen3.6 is actually much more stable with tools than Gemma 4.
2
mindinpanic
•
3d ago

Promising! Did you get any issues with the coding agent context?
1
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

Nope 😄
2
u/feik696 avatar
feik696
•
3d ago

I'm not too experienced with PCs, so I've mostly been using LM Studio, which has the same graphics card as yours. However, where LM Studio shows 30 tokens per second, I'm getting half that amount here. It's possible that I've made a mistake with the compilation, but then again, it wouldn't have started in the first place, right?
1
u/feik696 avatar
feik696
•
3d ago

code_python pred= 192 draft= 132 acc= 125 rate=0.947 tok/s=14.1

code_cpp pred= 192 draft= 138 acc= 121 rate=0.877 tok/s=13.4

explain_concept pred= 192 draft= 152 acc= 114 rate=0.750 tok/s=13.2

summarize pred= 53 draft= 40 acc= 32 rate=0.800 tok/s=14.1

qa_factual pred= 192 draft= 140 acc= 121 rate=0.864 tok/s=14.7

translation pred= 22 draft= 16 acc= 13 rate=0.812 tok/s=14.7

creative_short pred= 192 draft= 156 acc= 113 rate=0.724 tok/s=13.1

stepwise_math pred= 192 draft= 140 acc= 121 rate=0.864 tok/s=14.5

long_code_review pred= 192 draft= 146 acc= 117 rate=0.801 tok/s=13.7

Aggregate: {

"n_requests": 9,

"total_predicted": 1419,

"total_draft": 1060,

"total_draft_accepted": 877,

"aggregate_accept_rate": 0.8274,

"wall_s_total": 113.26

}
1
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

I also started with LM Studio, but to be frank, I never got good results with it. When I switched to llama.cpp, it was a night and day difference. LM Studio is a wrapper around llama.cpp that seems to add latency to the process. And you can never really be sure which parameters it passes to llama.cpp. If you can run llama.cpp directly, I'm pretty confident you'll get much better tok/sec!
1
ItsRektTime
•
3d ago

I got the following benchmark results on a 3060 12GB and R5 5600 with 32GB RAM:

// python3 mtp-bench.py
  code_python        pred= 192 draft= 148 acc= 116 rate=0.784 tok/s=40.3
  code_cpp           pred=  58 draft=  40 acc=  37 rate=0.925 tok/s=49.3
  explain_concept    pred= 192 draft= 148 acc= 116 rate=0.784 tok/s=41.6
  summarize          pred=  53 draft=  40 acc=  32 rate=0.800 tok/s=44.4
  qa_factual         pred= 192 draft= 144 acc= 119 rate=0.826 tok/s=45.6
  translation        pred=  22 draft=  16 acc=  13 rate=0.812 tok/s=40.8
  creative_short     pred= 192 draft= 166 acc= 108 rate=0.651 tok/s=38.1
  stepwise_math      pred= 192 draft= 138 acc= 122 rate=0.884 tok/s=46.7
  long_code_review   pred= 192 draft= 146 acc= 118 rate=0.808 tok/s=43.5

Aggregate: {
  "n_requests": 9,
  "total_predicted": 1285,
  "total_draft": 986,
  "total_draft_accepted": 781,
  "aggregate_accept_rate": 0.7921,
  "wall_s_total": 36.04
}

Also, I ran with -fitt 1736, since I use the 3060 as the primary GPU
1
u/EmelineRawr avatar
EmelineRawr
•
3d ago

Interesting, I also have a 4070 SUPER and was happy with a 40 tk/sec, I'll try your thing, thanks!!
1
u/OsmanthusBloom avatar
OsmanthusBloom
•
3d ago
• Edited 3d ago

Thanks a lot, this is inspiring! I'm trying to see if I can use MTP on my poor 3060 Laptop with just 6GB VRAM.

One stupid question though: how did you get mtp-bench.py working with current llama-server? What command did you use to run it?

For me it just gives 400 Bad Request errors regardless of how I try to run it. I suspect the problem is the call to "/completion" (I think it should be "/v1/completions"?)

EDIT: Nevermind, I found the problem. I was using llama-server with --models-preset, as I'm used to. But apparently it doesn't provide the exact same API that way, so the mtp-bench.py didn't work. I switched to running llama-server with separate CLI options and now it works!
1
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

You shouldn't have to pass anything to mtp-bench.py. It just starts, connects to your server and runs the benchmark.

If you look at the end of mtp-bench.py, you see the following line:

ap.add_argument("--url", default="http://127.0.0.1:8080")

If your server is not already running on http://127.0.0.1:8080, you can either modify mtp-bench.py to match your server host/port, or change your server port to match mtp-bench.py, and it should work 😄
1
u/OsmanthusBloom avatar
OsmanthusBloom
•
3d ago

Yeah. My problem was that I was using llama-server with the --models-preset option, which means it will run a proxy server on port 8080 and start separate workers for the requested model. In this mode the REST API is more limited and mtp-bench didn't work. As soon as I switched to the traditional CLI mode (lots of cli options) mtp-bench started working without any options.
2
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

Awesome! Glad you found the issue :)
2
u/Due_Steak_1249 avatar
Due_Steak_1249
•
3d ago

Have you observed any performance degradation as the context window reaches capacity? Historically, a 32k token limit appeared to be the optimal threshold for maintaining accuracy; for instance, Qwen3 reportedly showed a decline from 95% to 75% accuracy when scaling toward 128k.

Conversely, some users suggest that operating significantly below the 128k mark may increase the model's susceptibility to repetitive loops. I am interested in the current state of the art regarding this architecture and your practical experiences using it. It appears that users are currently forced to balance significant trade-offs between context volume and output reliability.
1
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

I've coded quite a bit with Qwen3.6, not as much with MTP though. Did lots of code additions, debugging and refactoring on ~10,000 line projects. Never noticed any degradation at all. Context was filling up quite fast though, so at 128K, I had to work on specific parts to prevent constant compaction.

Unfortunately, I realized Qwen3.6 cannot compete against larger models like GPT 5.5 for more demanding coding tasks, and often simply cannot produce any working code. But I still feel like Qwen is very capable for small projects where logic isn't pushed too far. I've had much more success with Qwen than Gemma 4.
1
u/leonbollerup avatar
leonbollerup
•
3d ago

Have you run any test to compare the quality against a “normal” model ?
1
Plastic_Use_4610
•
2d ago

Seems really high for the hardware - well done
1
u/the_masel avatar
the_masel
•
2d ago
• Edited 2d ago

Interesting, thank you. Did you compare it without MTP? With my 5060 Ti 16GB, I get around +15% tok/s and up to 66tok/s. Is this normal? (Tested on Windows 11)
1
u/Weird_Night_2176 avatar
Weird_Night_2176
•
2d ago

Been self-hosting AI for the past few months and finally got it to a point worth sharing. The stack:

- Jetson Orin Nano Super: CrewAI orchestration, 14 AI agents

- Orange Pi 5 Plus: Ollama model server

- Odroid XU4: PostgreSQL memory layer

- Jetson Nano 4GB: Tailscale mesh, network services

Total monthly cost: $8 (electricity + Claude API for final decisions only) The agents run a paper trading desk, generate SEO content for a local business client, write YouTube scripts, and send me a morning briefing every day via WhatsApp. All local, all private, zero cloud dependency.

Documenting the whole build on YouTube if anyone wants to follow along: https://www.youtube.com/@BlackBoxAILab

Happy to answer questions about the hardware setup or the agent architecture.
1
PeteInBrissie
•
2d ago

I’ve done this today and for some reason OpenCode is looping weirdly compared to the non-MTP setup. If I work it out I’ll share here
1
PeteInBrissie
•
1d ago

OK My setup is R5 5060G, 64GB RAM, RTX4060Ti. In OpenCode it was looping like mad until I set my context to 65576. Unfortunately OpenCode is also pushing 18,000 tokens at it which means an initial reaction time of about 3 minutes - after which it's really quick. Pretty sure I was seeing 90t/s at one stage last night.
1
zabadey
•
2d ago

Sorry for my dumb question, but does it mean that I can also use it with my 16gb ram mbp m5?
1
u/Snoo40301 avatar
Snoo40301
•
2d ago

Is this using the official llama.cpp or a fork for the MTP ?
1
u/trialbuterror avatar
trialbuterror
•
1d ago

Will this work for 9060xt 16gb 16gb ddr4 5600g processor ?

How effective is coding softwares ?
1
Resident_Worker_5807
•
1d ago
• Edited 1d ago

can i run it on Windows + Vulkan?
gpu is 4070 12gvram

32g ram on DDR4
1
Loouiz
•
17h ago

I've been running your config with a 16gb 4080 super, 7800x3d, 32gb ram. It is amazing, but I still get an occasional oom here and there. Any tips?
1
u/janvitos avatar
janvitos
OP •
10h ago
emoji:Discord:

Raise -fitt to something higher. Try 128 increments. If you're using 1536, try 1664 😄
1
u/leonbollerup avatar
leonbollerup
•
12h ago

Sadly.. the quality in the answer... goes to hell.. atleast in tests:
-- 

This is the prompt:
---
A city is planning to replace its diesel bus fleet with electric buses over the next 10 years. The city currently operates 120 buses, each driving an average of 220 km per day. A diesel bus consumes 0.38 liters of fuel per km, while an electric bus consumes 1.4 kWh per km.

Instructions:

    Verify your data

    Use tables to represent data where you can

Relevant data:

- Diesel emits 2.68 kg CO₂ per liter.

- Electricity grid emissions currently average 120 g CO₂ per kWh, but are expected to decrease by 5% per year due to renewable expansion.

- Each electric bus battery has a capacity of 420 kWh, but only 85% is usable to preserve battery life.

- Charging stations can deliver 150 kW, and buses are available for charging only 6 hours per night.

- The city's depot can support a maximum simultaneous charging load of 3.6 MW unless grid upgrades are made.

- Electric buses cost $720,000 each; diesel buses cost $310,000 each.

- Annual maintenance costs are $28,000 per diesel bus and $18,000 per electric bus.

- Diesel costs $1.65 per liter; electricity costs $0.14 per kWh.

- Bus batteries need replacement after 8 years at a cost of $140,000 per bus.

- Assume a discount rate of 6% annually.

Tasks:

    Determine whether the current charging infrastructure can support replacing all 120 buses with electric buses without changing schedules.

    Calculate the annual CO₂ emissions for the diesel fleet today versus a fully electric fleet today.

    Project cumulative CO₂ emissions for both fleets over 10 years, accounting for the electricity grid getting cleaner each year.

    Compare the total cost of ownership over 10 years for keeping diesel buses versus switching all buses to electric, including purchase, fuel/energy, maintenance, and battery replacement, discounted to present value.

    Recommend whether the city should electrify immediately, phase in gradually, or delay, and justify the answer using both operational and financial evidence.

    Identify at least three assumptions in the model that could significantly change the conclusion.

---

Result:
Comment Image
1
u/leonbollerup avatar
leonbollerup
•
12h ago
Comment Image
1
u/Creative-Type9411 avatar
Creative-Type9411
•
8h ago

the guide link is missing? for the "You can find a very nice guide on how to do that here and also download the..."??
1
singlegpu
•
3d ago

Any recommendations on where to learn more about this parameters?
1
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

Here you go: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

You can also ask any AI to explain them more in detail 😄 I got some pretty good answers from Gemini.
2
u/iamapizza avatar
iamapizza
•
3d ago

Does it work if you use --fit, --fit-target, and --fit-ctx? Supposedly these args should be taking care of using as much vram as possible.
1
u/unrevealedpains avatar
unrevealedpains
•
3d ago

how would It run on my 4GB VRAM, RTX 3050? I know this might be a stupid question but I am new to all of this
1
u/janvitos avatar
janvitos
OP •
3d ago
emoji:Discord:

Not stupid at all! You should try it 😄 I'd be curious and happy to see the result!
1
u/IrisColt avatar
IrisColt
•
3d ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Thanks a lot!!! 
1
evilbarron2
•
3d ago

Hmm. I get 100+ tok/sec (as measured by the llama-serve WebUI) with qwen3.6 35b A3b on my 3090 with my prompt.

---

Go to LocalLLaMA
r/LocalLLaMA
•
2d ago
CodProfessional3712
Speeding up local LLM for usable coding agent
Question | Help

TL;DR: Qwen 3.6 35B-A3B (Q4_K_M) is running slow at around 9 t/s with 72% filled context (36147 tokens window) and a total response time of 77s including prefill and token generation. Ran this using LM Studio on Windows with the attached image settings, on a 5060 Ti (16GB VRAM) + 32GB system RAM. I don't consider this performance great for my planned coding agent use case, so how can I speed this up? If I can't meaningfully speed it up, what other still-useful, faster LLMs do you suggest for my hardware specs?

Hello! As I see prices becoming tighter around cloud LLMs, I decided to look into local AI coding as a backup in case of a cloud LLM "apocalypse" or whenever I need to work with critical private software (I'm aware AI coding agents shouldn't be completely trusted around such things, I know the precautions to take).

I have a 5060 Ti (16GB VRAM) + 32 GB system RAM. To test if my hardware is capable of hosting a competitive local AI, I decided to load Qwen 3.6 35B-A3B into LM Studio, which uses a llama.cpp backend. Loading it with around 32K context window, it runs at a decent speed of 17 t/s with just a simple "Hi" prompt. However, if you've used coding agents before, you'll know they often come with a hefty system prompt on top of the code that's shoved into the context window, so I need to test if the LLM is usable at high context load.

I used 4-bit quantization for KV cache, why? I've read online that TurboQuant's speed advantage is not too different from 4-bit KV Cache quantization (space gains are very much real though), so I decided to triage that first using LM Studio's easy setup. I gave it a chunk of Frankenstein's text from Project Gutenberg to fill its context to 72%, it took 77s to generate a response, with a decent chunk of it being in the "Processing" of the prompt (I assume this is the "prefill", which comes before token generation itself). Token generation speed was 9 t/s.

The issue here is that speed is obviously not the best, which does not bode well for coding agents, where you're meant to iterate quickly. Better to fail fast with less capable agents so you can steer them better while knowing their limitations.

I was wondering if you could give me insight into how to speed up this LLM or if this version of Qwen is simply out of the league for my hardware specs. If it's out of my league, what usable coding LLMs would you recommend for my hardware? I know "usable" may not be specific, so I mean something like 90%-80% of what cloud agents can do or at the very least what the Qwen model I already tested can do.

For more details on how I'm running this particular model, see the image I've attached. It's my LM Studio configuration, not exactly a terminal command setup. If running the llama.cpp backend without the LM Studio frontend offers a better speed-up, please let me know! I'm running this on Windows.
r/LocalLLaMA - LM Studio config, Windows, 5060 Ti (16GB VRAM) + 32 GB RAM
LM Studio config, Windows, 5060 Ti (16GB VRAM) + 32 GB RAM
22
u/decartai avatar decartai
•
Promoted
If you could be anyone on cam, who would you be?
Download
delulu.cam
Clickable image which will reveal the video player: If you could be anyone on cam, who would you be?
Sort by:
Comments Section
theUmo
•
2d ago

I have a similar setup to yours (same video card with 32 gb of ram) and my baseline speed is closer to 35-40 t/s using llama.cpp / llama-swap with these parameters:

  "qwen3.6-35b-a3b":
    cmd: |
      llama-server
        --port ${PORT}
        --webui-mcp-proxy
        -c 65536
        -ctk f16
        -ctv f16
        --fit on
        -fa on
        -t 8
        -b 2048
        --ubatch-size 512
        --presence-penalty 0.0
        --repeat-last-n 128
        --jinja
        --image-min-tokens 1024
        --alias qwen3.6-35b-a3b
        -m "F:/AI/Models/LLM/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
        --mmproj "F:/AI/Models/LLM/mmproj-Qwen3.6-35B-A3B-BF16.gguf"
        --chat-template-kwargs "{\"preserve_thinking\":true}"
        --reasoning on

15
u/CodProfessional3712 avatar
CodProfessional3712
OP •
2d ago

Thanks for sharing, I'll try this out. Are you using any particular chat frontends?
3
theUmo
•
2d ago

By the way, if you take the output from the llama.cpp loading process and ask Qwen to analyze it, you might be surprised at how well it sifts through it to identify issues and opportunities for optimization. FlashAttention not loading when it should, layers not getting offloaded, etc.

It also really helps to have process explorer running to verify vram is getting used and the gpu is busy, if you're not already monitoring that.
8
u/CodProfessional3712 avatar
CodProfessional3712
OP •
2d ago

Thanks for the tip!
2
theUmo
•
2d ago

Nope, just testing from the web interface that llama-server spins up.

Also, you don't need llama-swap to test this; just get the latest llama.cpp, assemble the commandline above (replacing ${PORT} with the actual port you want) and it should work.

The key for me here was moving away from LM Studio. At the time I made the switch, llama.cpp had some upstream improvements for Qwen and their llama.cpp version was at least a month out of date and used a version number that didn't clearly correspond to a specific llama.cpp build.
6
u/GlobalLadder9461 avatar
GlobalLadder9461
•
2d ago

What does webui mcp proxy does?
1
theUmo
•
2d ago

It makes it so that some mcp tool calling setups can work correctly.

My understanding is that it enables CORS for some of the internal communications, and whether it's needed depends on whether you're using localhost for everything in your mcp setup.

If you're not trying to get the web chat interface of llama.cpp to use tools over mcp, you don't need it at all.
1
u/ihateuall18 avatar
ihateuall18
•
2d ago

I'm not an expert, but I had the same experience as you on Windows. Then I tried LM studio on linux, got double the t/s gen, then tried llama.cpp, I am getting 40-50 t/s up to on a 2070s. I'd say you try llama.cpp + cuda on windows, if you are not getting anything close to what I am getting, make the switch..

llama-server -m /path/to/model/unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q6_K --fit on --temp 1.0 --jinja --top-p 0.95 --top-k 20 --presence-penalty 1.5 --min-p 0.0 -c 120000 --n-cpu-moe 38 --no-mmap -ctk q8_0 -ctv q8_0 -fa on --chat-template-kwargs '{"preserve_thinking":true}'

you can play with the context, n-cpu-moe change to 999, add --no-mmproj (removes vision from model I believe), etc. I'm not a pro just what worked for me, it really made a difference.
5
u/Youknowwhyimherexxx avatar
Youknowwhyimherexxx
•
2d ago

When a model can’t fit in your vram it needs to move information to and from the cpu, this is what causes the big slow down.

Maybe try moving batch to 512, try offloading more model to gpu - also try forcing some of the moe weights onto cpu (helps for this kind of model).
there’s always moving the quant down too
4
u/Meshyai avatar u/Meshyai
•
Promoted
🔥 Up to 70% OFF all plans. 500 bonus credits on Pro Monthly. Biggest sale in Meshy history. Claim now.

    https://www.meshy.ai/anniversary-2026
    https://www.meshy.ai/anniversary-2026
    https://www.meshy.ai/anniversary-2026

meshy.ai
Sign Up
u/kobraca avatar
kobraca
•
2d ago

Im running 50-52tps with 12gb vram 4070 super and 32gb ram, 131k context in q8 cache, something is off by a lot in your config. I will be at my pc in an hour and share my config
4
u/CodProfessional3712 avatar
CodProfessional3712
OP •
2d ago

I'm open to hearing your feedback about the config!
1
u/kobraca avatar
kobraca
•
2d ago

llama-server -m Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --host 127.0.0.1 --port 1234 -c 128000 -ctk q8_0 -ctv q8_0 -fa on -ngl 999 --n-cpu-moe 26 -t 6 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.05 --chat-template-kwargs "{\"preserve_thinking\": true}" --reasoning-budget 4096

prompt eval time = 423.38 ms / 50 tokens ( 8.47 ms per token, 118.10 tokens per second)

eval time = 2558.36 ms / 135 tokens ( 18.95 ms per token, 52.77 tokens per second)
2
u/CodProfessional3712 avatar
CodProfessional3712
OP •
17h ago

Thanks!
2
MrShrek69
•
2d ago

Context always falls apart over 65k try breaking up ur prompts into smaller stuff that could be finished in less token output. Use open code or pi-mono
3
u/dtrq avatar
dtrq
•
2d ago

Running this model on 12gb VRAM + 16gb RAM with ~35 tps. Seet 'GPU iffload' full on, then set 'number of MoE layers to force on CPU' (bottom slider) to something between 1/3 to half. This will run heavier stuff on GPU while keeping less demanding layers in RAM.
2
u/Telethex avatar
Telethex
•
2d ago

Using the same model and quant, I'm getting 21 tok/sec at 112k/124k context filled on llama.cpp.. you can definitely do better on that card as I'm using a Radeon 6900xt 16gb which can't compare. I'm not using MTP or turboquant either. Keep messing with it!
2
u/Shaped_ai avatar u/Shaped_ai
•
Promoted
Stop drowning your local weights in 50k tokens of RAG noise.
u/elongated-muskmelon avatar
elongated-muskmelon
•
2d ago

I have the exact same setup as you, and i get around 40-45 tps with maxed out context. I use turboquants though. I’m not near my workstation right now, will share the launch command once i get home.
2
u/DiscipleofDeceit666 avatar
DiscipleofDeceit666
•
2d ago

Number of experts to force on the cpu

Increase that number.

The way this works is that the model has several experts, you make a request, and only a few of those experts are working and the rest are not. Keeping some of the useless experts in RAM frees up your GPU for kvcache calculations.

Probably the only speedup id chase here, you’re paging ram.
3
u/rlobo avatar
rlobo
•
2d ago

This. Max out GPU usage and MoE to CPU and then start optimizing from there.
2
Xantrk
•
2d ago

Offload "all" layers to GPU (this is actually only dense layers), and start from 35 Number of layers to offload to CPU and walk backwards based on available RAM and performance cliff.

Highly recommend ditching LM Studio for LLama.cpp or Jan (if you need a GUI). It uses old llama.cpp and lags far behind optimizations.

For your benchmark, with 12 gb VRAM, 32 GB system RAM, I get 800 tk/s promp processing and 40-50 tk/s generation with 100k context.

For MOE models the trick is to have all layers in GPU, and offload only overflowing MOEs to CPU. Llama.cpp does this automatically for you
2
jake_that_dude
•
2d ago

I'd separate prefill from decode before changing models. At 36k tokens, your bad number is probably prompt processing more than the 9 t/s decode.

Run the same packed prompt through current llama.cpp with -fa on, --ubatch-size 512, and VRAM/process explorer open. If prefill dominates, shrinking the agent context window will feel faster than swapping to a smaller model.
1
Life-Screen-9923
•
2d ago

You should try to change number "GPU offload", try in range 25-29

Test every number, don't change context size while testing
1

---

Go to LocalLLaMA
r/LocalLLaMA
•
7h ago
coder543
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter
Drastically improve prompt processing speed for --n-cpu-moe partially offloaded models
Tutorial | Guide
Bigger ubatch made gpt-oss-120b prompt processing much faster on my RTX 3090

I was tuning gpt-oss-120b-F16.gguf with llama.cpp on a 24 GB RTX 3090 and found that increasing the physical micro-batch size (-ub) can massively improve prompt processing throughput, as long as you also raise --n-cpu-moe enough to keep the run inside VRAM.

The llama.cpp defaults are -b 2048 and -ub 512; I included that default run as its own point in the chart.

Here are the informal llama-bench results I charted:
ubatch 	n-cpu-moe 	prefill 	generation
256 	25 	240.03 tok/s 	33.14 tok/s
512 (default) 	26 	380.27 tok/s 	32.29 tok/s
2048 	25 	1112.54 tok/s 	32.96 tok/s
4096 	26 	1682.47 tok/s 	32.38 tok/s
8192 	28 	2090.68 tok/s 	30.05 tok/s

Compared with the llama.cpp default -ub 512, prompt processing went from about 380 tok/s to about 2091 tok/s, roughly a 5.5x gain. Compared with the smaller -ub 256 run, it was about an 8.7x gain. Token generation dropped from about 32.3 tok/s at default settings to 30.1 tok/s at -ub 8192, about a 7% reduction.

The catch is that the larger ubatch needs more GPU compute workspace. On my machine, -ub 4096 needed --n-cpu-moe 26, and -ub 8192 needed --n-cpu-moe 28. So this is a throughput trade: move a few more MoE layers to CPU to make enough room for the bigger batch, and prompt-heavy workloads get dramatically faster while generation gets a little slower.
r/LocalLLaMA - Drastically improve prompt processing speed for --n-cpu-moe partially offloaded models

Note: the first four prefill points are pp4096; the 8192 ubatch point is from a pp8192 run, so treat this as an informal tuning result rather than a perfectly controlled benchmark.

-----

One of the reasons I bought a DGX Spark was to have better prompt processing speeds. If I had known about this trick, I might not have done that in retrospect, even though it is a very nice machine, and still gets slightly better prompt processing performance and like double the token generation speed for gpt-oss-120b. Higher ubatch drastically closes the gap.
57
u/Vladi-N avatar Vladi-N
•
Promoted
Four Divine Abidings: a hand-painted idle game that helps slow down. Calm. Relax. Let go
Play Now
store.steampowered.com
Thumbnail image: Four Divine Abidings: a hand-painted idle game that helps slow down. Calm. Relax. Let go
Sort by:
Comments Section
u/ikkiho avatar
ikkiho
•
5h ago

fwiw the reason -ub helps so much here is that with --n-cpu-moe your attention and router still run on the 3090 and those are the launch-overhead bound kernels during prefill. bigger ubatch means fewer kernel launches per chunk so the GPU stays saturated. generation doesn't move because that's one token at a time, you're memory-bandwidth bound on the CPU expert weights and that part doesn't care about -ub at all. nice writeup, this trick is buried in the llama.cpp issues.
11
u/notdba avatar
notdba
•
3h ago

Err, I don't think that's right. Bigger ubatch means you amortize the PCIe transfer overhead across more tokens. Check out the math in https://github.com/ikawrakow/ik_llama.cpp/pull/520
4
u/End0rphinJunkie avatar
End0rphinJunkie
•
2h ago

Spot on, it's easy to forget how much CPU overhead from kernel scheduling drags down prefill if your not batching enough. Definetly feels like a trick that should be pulled out of the github issues and put in the main docs.
2
u/draconds avatar
draconds
•
7h ago

You are a legend, sir!
This was the only thing that helped me.
Everything else just said turn flash attention on.
I was only using the -b flag, but as soon as i increased -ub, it became ideal.
Thank you for your service!!!
5
u/OsmanthusBloom avatar
OsmanthusBloom
•
4h ago

Thanks for the excellent and detailed writeup. I discovered the same thing a while ago (increasing ubatch size can drastically improve PP speeds for partially offloaded MoE models at the cost of some TG speed) and I've been trying to spread the word in some comments. But of course such comments deep down the threads are only seen by relatively few people.

Some of my bench results showing effect of ubatch size: https://www.reddit.com/r/LocalLLaMA/comments/1rg4zqv/comment/o7rszuj/

Other comments of mine with this advice e.g.:

https://www.reddit.com/r/LocalLLaMA/comments/1rg4zqv/comment/o7r3zka/

https://www.reddit.com/r/LocalLLaMA/comments/1rgkmd7/comment/o7uq292/

https://www.reddit.com/r/LocalLLaMA/comments/1rh9983/comment/o7xcemx/

https://www.reddit.com/r/LocalLLaMA/comments/1rz43hi/comment/objvubg/

https://www.reddit.com/r/LocalLLaMA/comments/1sprdm8/comment/oh3ulwt/
3
u/Snoo_81913 avatar
Snoo_81913
•
5h ago
• Edited 5h ago

I mean the default -ub is set at 512 because it's a safe number to keep cards with lower amounts of VRAM from having memory spikes. If you have the VRAM you can adjust until you hit the saturation or VRAM limits. Once you're saturated the benefits stop and if you hit VRAM the dreaded OOM.

The baseline is set so there isn't a million reddit posts saying "Llama is GARBAGE all I get is OOM!" LMAO.

There can also be thermal throttling with larger batch sizes, though this is mainly a unified memory issue.

I only have an 8gb card and I ride the line so I always run 2048/512 on my models that take up 6gb+ and 2048/2048 on small models if it makes sense

Nice work though, I like to see posts with real test data.
4
u/coder543 avatar
coder543
OP •
5h ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

    The baseline is set so there isn't a million reddit posts saying "Llama is GARBAGE all I get is OOM!" LMAO.

No... it would just OOM on startup and you'd select a lower amount of context, as you already do today. It would not be any different other than how much context you could fit by default without lowering ubatch manually. Most users these days probably use -fit (which is enabled by default) and don't specify a context at all, just getting whatever llama thinks will fit.

The bigger issue for most users is that llama-server defaults to --parallel 4, which allocates 4 KV caches when most users only need 1. Setting --parallel 1 easily lets you fit more context onto your GPU's limited VRAM.

I also never said the default should be something else, although I think it should be higher for -ncmoe users by default, since -ncmoe suffers significantly more from smaller ubatches.
4
u/reacusn avatar
reacusn
•
2h ago

    defaults to --parallel 4

Does it actually?
From my experiences building rocm and cuda on debian 13 in the past year (haven't check older versions), as well as downloading prebuilt windows binaries for cuda and vulkan, llama-server defaults parallel to to -1, which allocates 4 slots (for my system, at least) but also enables unified kv cache if it's not explicitly set, dynamically reducing context as as required while taking up the same memory use as np1.

https://github.com/ggml-org/llama.cpp/pull/17997:
	
--kv-unified, -kvu 	use single unified KV buffer shared across all sequences (default: enabled if number of slots is auto)
-np, --parallel N 	number of server slots (default: -1, -1 = auto)
	
2
u/Snoo_81913 avatar
Snoo_81913
•
4h ago

--fit is just pure laziness lol. Don't get me wrong sir I'm definitely going to test the limits using your data.

I wasn't aware Llama defaulted to - - parallel 4 I run -np 1 which is the same as - -parallel 1 correct?

I'm running cmoe 35 right now, what do you think of -ot? Actually here's my fastest set up with a 4060 is there room for improvement here? I've been tweaking it. I run it Q8/Q4 or Q4/Q4. Works pretty good either way. Slightly slower than 4/q4. I am getting 40 t/s and I'd have to look it up and I wouldn't swear on it but I'm pretty sure I saw over 1k prefill in my log when I was testing.

``bash llama-server -m Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf
-c 196608 -ngl 99
--n-cpu-moe 35 --no-mmap --flash-attn on
-b 2048 -ub 512
--cache-type-k q4_0 --cache-type-v q4_0
--reasoning-budget 4096 --cache-ram 0 -np 1 -t 6
--port 8080

-1
u/QVAC_Official avatar u/QVAC_Official
•
Promoted
Why pay per token or per minute when you can run it on your own hardware? QVAC SDK is built for developers who want to cut the cord from centralized cloud providers. It's fully open-source, scalable, and engineered to run anywhere without dependency on an external server.
Learn More
docs.qvac.tether.io
Thumbnail image: Why pay per token or per minute when you can run it on your own hardware? QVAC SDK is built for developers who want to cut the cord from centralized cloud providers. It's fully open-source, scalable, and engineered to run anywhere without dependency on an external server.
jacek2023
•
7h ago
llama.cpp
Profile Badge for the Achievement Top 1% Poster Top 1% Poster

I use this right now on 3x3090:

./bin/llama-server -c 200000 -m /mnt/models2/Qwen/3.6/Qwen3.6-27B-UD-Q8_K_XL.gguf --host 0.0.0.0 --jinja -fa on --keep 4096 -b 8192 --spec-type ngram-mod --parallel 1 --ctx-checkpoints 24 --checkpoint-every-n-tokens 8192 --cache-ram 65536 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 --repeat-penalty 1.0

I assume you know you can run llama-bench with multiple values to produce all results on one run?
2
u/coder543 avatar
coder543
OP •
7h ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

You only appear to be changing your batch size, not ubatch, but since you have 3x 3090, you could fit the entire GPT-OSS-120B model into VRAM anyways, so the benefits may not be much. The same applies for the even smaller 27B model you’re using, of course.

Most of the benefit here is that a larger ubatch reduces the number of times the weights have to be streamed from slow CPU RAM during prefill.
4
overand
•
3h ago

I believe that  UD ..K_XL actually came out a little bit worse than the Q8_0 - you should look at kl divergence graphs. You might be able to save a little bit of VRAM while also getting a slightly more accurate quant
1
FullstackSensei
•
1h ago
llama.cpp
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Last I checked K_XL had better KL divergence in Unsloth's own tests
1
u/AdventurousFly4909 avatar
AdventurousFly4909
•
6h ago

What cpu do you have then
3
u/coder543 avatar
coder543
OP •
6h ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

This machine is using a Ryzen 9 7950X, but prefill batches are streamed to the GPU, so it shouldn’t make much difference, I’d imagine.
3
u/Wise-Hunt7815 avatar
Wise-Hunt7815
•
6h ago

That's right, it's just a pity that it consumed too much vram.
2
u/Fast-Satisfaction482 avatar
Fast-Satisfaction482
•
2h ago

Does this generalize to other models? Can we improve prompt processing with this trick as long as we have spare VRAM? 
1
StorageHungry8380
•
56m ago

Generalize as in the llama.cpp defaults are probably not optimal, yes. Generalize as in larger microbatch == better, no. As an example, Qwen3.6 27B Q5 128k context on a 5090 running Windows 11, I benchmarked that having -b 1024 -ub 1024 was optimal. So I'd say spend some time with llama-bench on your specific scenarios.

---


Go to LocalLLaMA
r/LocalLLaMA
•
28d ago
raketenkater
The LLM tunes its own llama.cpp flags (+54% tok/s on Qwen3.5-27B)
Resources

This is V2 of my previous post.

What's new: --ai-tune — the model starts tuning its own flags in a loop and caches the fastest config it finds.

My weird rig: 3090 Ti + 4070 + 3060 + 128GB RAM.
Model 	llama-server 	llm-server v1 tuning 	llm-server v2 (ai-tuning)
Qwen3.5-122B 	4.1 tok/s 	11.2 tok/s 	17.47 tok/s
Qwen3.5-27B Q4_K_M 	18.5 tok/s 	25.94 tok/s 	40.05 tok/s
gemma-4-31B UD-Q4_K_XL 	14.2 tok/s 	23.17 tok/s 	24.77 tok/s

What I think is best here: --ai-tune keeps up with updates on llama.cpp / ik_llama.cpp automatically, because it feeds llama-server --help into the LLM tuning loop as context. New flags land → the tuner can use them → you get the best performance.

i think those are some solid gains (max tokens yeaaahh), plus more stability and a nice TUI via llm-server-gui.

Check it out: https://github.com/raketenkater/llm-server
177
u/scrapedo_ avatar scrapedo_
•
Promoted
No more blocks, no more scraper downtime. Meet the better, faster, stronger web scraping API. Scrape.do beats competition every time.
Learn More
scrape.do
Thumbnail image: No more blocks, no more scraper downtime. Meet the better, faster, stronger web scraping API. Scrape.do beats competition every time.
Sort by:
Comments Section
u/segmond avatar
segmond
•
28d ago
llama.cpp
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

provide an example of the parameters it used vs the previous to go from 4.1tk/s to 17.47tk/s
48
u/raketenkater avatar
raketenkater
OP •
28d ago
• Edited 28d ago

so 4.1tk/s is just

llama-server -m Qwen3.5-122B-A10B-Opus-Reasoning-Q4_K_M.gguf

and the tuned command after moe expert placment which layer 1 handles. And ai-tune (layer 2 optional ) this would be

llama-server -m Qwen3.5-122B-A10B-Opus-Reasoning-Q4_K_M.gguf \
-ngl 48 \
--tensor-split 0.54,0.23,0.23 \
-sm graph \
-fa on \
--cache-type-k q8_0 --cache-type-v q8_0 \
-ot "blk\.(1[4-9]|2[0-9])\.ffn_.*_exps=CUDA1" \
-ot "blk\.(3[0-9]|4[0-7])\.ffn_.*_exps=CUDA2" \
--run-time-repack -khad --defrag-thold 0.1 \
--threads 8 --threads-batch 16 \
--batch-size 2048 --ubatch-size 256

Gets 17.47 tok/s
26
ecompanda
•
28d ago

the cpu offload strategy being the default when ngl is not set explains a lot of the bad benchmarks people post. most "my llama.cpp is slow" threads are just missing that one flag
31
Liquos
•
28d ago

I thought in the latest versions it defaults to offloading to the GPU?
6
StardockEngineer
•
28d ago
vllm
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

It does. llama.cpp defaults to maxing out the GPU
10
u/IrisColt avatar
IrisColt
•
27d ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

ngl... ngl is a must
3
u/Equivalent_Job_2257 avatar
Equivalent_Job_2257
•
28d ago

Can you compare with the same command as first with only '--fit' flag added?
2
u/666666thats6sixes avatar
666666thats6sixes
•
27d ago

--fit is on by default unless they used some ancient (pre 75xx) build
1
u/draetheus avatar
draetheus
•
28d ago

I have 96GB ram and a single 9070 XT and with vulkan I get about the same TG speed with the same quant. What is your PP speed though? If you have any MoE layer spilling onto CPU, ubatch size of 256 is going to be horrible for PP speed. I'm not sure I trust this as the most optimized settings possible.

I would honestly get a baseline with just your 3090 TI and CPU offload, you might be starved for PCIe bandwidth trying to split across that many GPUs.
1
u/raketenkater avatar
raketenkater
OP •
28d ago

yes my 3060 is currently only 1x but i will upgrade to 4x using m.2 to pcie adapter soon hehe
1
u/Glittering-Call8746 avatar
Glittering-Call8746
•
28d ago

Will it work for ik_llama.cpp ?
1
u/raketenkater avatar
raketenkater
OP •
27d ago

yes but ik_llama.cpp is often unstable but faster
1
ForsookComparison
•
28d ago
emoji:Discord:
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

i think they meant the before and after
0
u/segmond avatar
segmond
•
28d ago
llama.cpp
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

I think they showed the before, they ran it without offloading to GPU
"llama-server -m Qwen3.5-122B-A10B-Opus-Reasoning-Q4_K_M.gguf"

To OP, at least offload to GPUs and use the fit parameters, that should be your minimal baseline.
24
StardockEngineer
•
28d ago
vllm
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

That would totally offload to the GPU. That's the default.
2
[deleted]
•
28d ago

Profile Badge for the Achievement Top 1% Commenter

Pixer---
•
28d ago

Will there be a rocm / Vulkan version ?
19
u/raketenkater avatar
raketenkater
OP •
28d ago

working on vulkan right now for v2.2
7
u/raketenkater avatar
raketenkater
OP •
23d ago

Vulkan support is here check it out latest commit https://github.com/raketenkater/llm-server/tree/vulkan
1
CornerLimits
•
28d ago

Maybe a simple script without llm could be faster/better no burned tokens? It will bench a lot of times, i can’ t see the real value of having an llm.

However cool idea!
9
u/raketenkater avatar
raketenkater
OP •
28d ago
• Edited 28d ago

There is no token burn it is using your local llm and you do not have to use the ai-tune it’s an optional flag then it is a simple script 🙂
7
CornerLimits
•
28d ago

I mean a token is a token even if its local :D Btw like this is certainly a good approach because you don’t have to update it if new flags comes out. It only sounds a bit overkill to me and i suspect low tire llms will make a mess with all the flags but maybe i’m wrong.

How can it detect when we are at absolute maximum performance (no stuff to try left) ?
4
u/raketenkater avatar
raketenkater
OP •
28d ago

even small current llms are pretty capable but i have not tested it and for the maximum performance ai-tune basically just uses the rounds you set and runs that often while crashes do not count and then checks which is the best tk/s and safes that but i think maximum performance is relativ with ai space llama.cpp moving so fast that its just going up
3
u/fishhf avatar
fishhf
•
28d ago

Using the optuna library or the old genetic algorithm would be less of an overkill.
1
IsopodInitial6766
•
28d ago
• Edited 28d ago

The value isn't faster search it's zero-maintenance search. Optuna needs you to define the parameter space up front: which flags exist, valid values, conflicts. An llm reading `llama-server --help` each run picks up new flags (like `--ubatch-size`) without updating your config Hybrid is probably best: LLM constrains the search space, a deterministic tuner does the sweep
3
u/Meshyai avatar u/Meshyai
•
Promoted
🔥 Up to 70% OFF all plans. 500 bonus credits on Pro Monthly. Biggest sale in Meshy history. Claim now.

    https://www.meshy.ai/anniversary-2026
    https://www.meshy.ai/anniversary-2026
    https://www.meshy.ai/anniversary-2026

meshy.ai
Sign Up
mister2d
•
28d ago

It's always nice to see optimization on consumer hardware. I've had to do this by hand while keeping up with all the new flags like n-cpu-moe and tensor parallelism.

And since buying a new rig is out of the question I have to squeeze out everything from my DDR3 box.
6
u/raketenkater avatar
raketenkater
OP •
28d ago

Exactly same for me
3
u/Glittering-Call8746 avatar
Glittering-Call8746
•
28d ago

So basically it keep trying till it get the right tensor split ?
4
u/raketenkater avatar
raketenkater
OP •
28d ago
• Edited 28d ago

there are 2 stages

1. it calculates vram pcie lane speed and model architecture and so on(system,model specs) based on that it chooses a strategy: 1. dense single gpu 2. dense multi gpu 3. cpu offloading expert placment(with first conservativ placment then filling it until tight)

2. there is an ai-tune flag which prompts the llm running with the --help of the selected backend and then trys to better its tk/s performance by for example better tensor split yes
12
RelicDerelict
•
28d ago
Orca

Fantastic, finally PCI lanes getting into consideration. Building PC with PCIe 5.0 doesn't sounds so useless anymore.
1
u/Glittering-Call8746 avatar
Glittering-Call8746
•
28d ago

No it doesn't until u figure out pcie 5.0 extension have to be shorter and are more expensive and not all consumer mobo have pcie 5.0 bifurcation..
1
u/Designer_Reaction551 avatar
Designer_Reaction551
•
28d ago

the self-tuning loop idea is actually brilliant for multi-GPU setups where the optimal layer split is basically impossible to guess manually. we spent hours tweaking ngl and tensor split values for a 3090 + 3060 combo before just writing a similar brute-force search. 4.1 -> 17.47 on the 122B is wild tho, most of that is probably just proper GPU offloading vs CPU default.
4
HopePupal
•
28d ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

brute force seems fine for multi-GPU setups you can count on one hand. you don't need an LLM for that. you don't even really need Optuna or another hyperparameter search tool.
2
u/interservermike avatar u/interservermike
•
Promoted
VPS Hosting For Self Hosted AI Tools.
Learn More
interserver.net
Thumbnail image: VPS Hosting For Self Hosted AI Tools.
u/ketosoy avatar
ketosoy
•
28d ago

Do you have a genetic algo in there or is it pure random testing?
3
u/fuchelio avatar
fuchelio
•
28d ago

Does --ai-tune support hard constraints? For example, a 256K context, mmproj, or thinking as a non-negotiable requirement.
3
u/raketenkater avatar
raketenkater
OP •
28d ago

yes ctx-size is set can not be changed by ai-tune as well as vision
3
Queasy_Asparagus69
•
28d ago

Wow I love this
2
u/b1231227 avatar
b1231227
•
28d ago

Can it export the parameters after ai-tune as a reference? Because I am using another llama.cpp branch, there are some functions that I need so I cannot directly jump to the llm-server you developed.
3
u/raketenkater avatar
raketenkater
OP •
28d ago

which binary to run is pluggable using the --server-bin flag too
3
u/raketenkater avatar
raketenkater
OP •
28d ago

It saves them as configs so yes
1
TomHale
•
28d ago

Very cool! With your AIs knowledge and context, could you ask if for a plan on how to do the same but with Lemonade for AMD?

A markdown file on that in your repo on that would be amazing! 😉
5
ai_without_borders
•
28d ago

tensor split is doing a lot of heavy lifting here. with mixed vram capacities (like 3090+4090), the default 50/50 split hammers the slower card and you get bottlenecked at the compute boundary. finding the right ratio is sometimes worth 2x on its own, separate from any flag tuning. curious what the split ended up being in the optimized config.
2
RelicDerelict
•
28d ago
Orca

Does this calculates ratio between CPU and GPU too?
1
ai_without_borders
•
26d ago

sort of. --tensor-split divides layers across gpus only. cpu offload is controlled by -ngl. layers outside that budget fall to cpu. you tune the gpu/cpu boundary by adjusting -ngl, not the split ratios.
1
RelicDerelict
•
26d ago
Orca

OK but what about offloading only FFN to CPU and RAM and attention layers to GPU? Because that is the most effective way how to saturate GPU and CPU at the same time (especially ik_llama is good at that), just to fiddling with the ratio is finnicky, or I misunderstood some concepts? Thanks!
1
ai_without_borders
•
24d ago

yeah ik_llama.cpp's --override-tensors is what enables this. regex patterns let you target by tensor name - push attn_q/k/v/o to GPU, keep ffn_gate/up/down in RAM. for MoE models it works better than you'd expect: expert FFNs are huge but sparse (not all active per token), so streaming from RAM has lower overhead than on dense models. standard -ngl doesn't give you that granularity - it's a layer count cutoff, not layer-type routing. ik is the right tool if you want the layer-type split.
1
andy2na
•
28d ago
llama.cpp
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

any easy way to run this in a docker container? I've tried to run it in unraid and its not working at all
2
u/rearwebpidgeon avatar
rearwebpidgeon
•
28d ago

Seems like --ai-tune isn't implemented in llm-server-mac - that wasn't clear to me from docs (unless I just didn't RTFM enough).
2
u/raketenkater avatar
raketenkater
OP •
27d ago

yeah sry about that will fix soon
2
u/amb007 avatar
amb007
•
27d ago

Also please add support for bash 3.2.57 (default). E.g.

sed -n '/^# /!d; s/^# //p' "$0" | grep -v "shellcheck"
1
ecompanda
•
28d ago

u/raketenkater avatar

u/raketenkater avatar

Theboyscampus
•
28d ago

It sounds like auto OCing on graphic cards lol
1
u/JLeonsarmiento avatar
JLeonsarmiento
•
28d ago
emoji:Discord:

What kind of witchcraft is this?
1
u/Danmoreng avatar
Danmoreng
•
28d ago

Have you tried optimal default settings with fit and fit-ctx? See here: https://github.com/Danmoreng/local-qwen3-coder-env
1
u/sonicnerd14 avatar
sonicnerd14
•
28d ago

Interesting, ironically I've been working on a skill that does something similar called local inference optimizer. Except that it relies on an agent outside of the LLM working on the host machine itself to find the most optimal settings. I think both ideas are pretty solid and useful so that we dont have to spend so much time tuning these models ourselves.
1
RelicDerelict
•
28d ago
• Edited 28d ago
Orca

Will be Linux supported in the future? Also does this use all optimization flags like override tensors, smart MoE pick, intelligently offloading FFNs to system ram and attention layers to GPU, arbitrary KV size performance, etc.?
1
u/unculturedperl avatar
unculturedperl
•
27d ago

It works on linux now.
1
u/Corosus avatar
Corosus
•
28d ago
• Edited 28d ago

Cool stuff, on a whim I decided to try it, bothering to switch from windows to wsl2 alone has given me a nice lil boost from 26tps to 30tps for Qwopus3.5-27B-v3-Q4_K_M.gguf, i really need to get back to bare metal linux to get the direct pcie to pcie communication, for now lets see if it can beat my current kinda mostly optimized ik_llama on my dual 5070ti 5060ti with bad pcie speed communication setup

~/projects/git/ik_llama.cpp/build/bin/llama-server -m /home/corosus/projects/ai/jackrong/Qwopus3.5-27B-v3-Q4_K_M.gguf --host 0.0.0.0 --port 8080 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.4 --repeat-penalty 1.5 -ngl 99 -sm layer --merge-qkv -rtr -fa on -ctk q8_0 -ctv q8_0 -c 100000 -b 16384 -ub 4096 -ts 28,20 --no-warmup --jinja --numa isolate -tb 16

at Round 4/8 so far on the --ai-tune

it more or less matched the speed I had already worked out

mine goes at about 29tps

it reported it was able to peak at 32tps

it then runs the server with these params:

/home/corosus/projects/git/ik_llama.cpp/build/bin/llama-server -m /home/corosus/projects/ai/jackrong/Qwopus3.5-27B-v3-Q4_K_M.gguf --host 0.0.0.0 --port 8080 --ctx-size 65536 --flash-attn on -b 4096 -ub 512 --cache-type-k q4_0 --cache-type-v q4_0 --jinja --threads 10 --threads-batch 10 --run-time-repack -khad --no-context-shift --defrag-thold 0.1 -mqkv -cram 1870 --ctx-checkpoints 9 -ngl 999 -mg 0 --tensor-split 0.7,0.3

hits about 27tps, but im dealing with some variability of wsl running in a well used windows machine

so i can more or less conclude it found the best settings for my setup, about the same as what i already had, within margin of error

should be handy for those not wanting to tweak for days and days and days.

I should give this a try for 122b to find a more tightly tuned moe offload strat.
1
u/unculturedperl avatar
unculturedperl
•
28d ago
• Edited 28d ago

For schnitts and giggles ran it on my dgx spork (n100/16gb/gemma4-e2b-2b q8_0).

###################################################
  AI Tune complete: Maximize KV Cache Quality and Batch Size wins!
  Baseline: 7.64 tok/s # Best: 7.75 tok/s (+1.4%)
###################################################

The changes it suggested were to use:

--cache-type-k q8_0 --cache-type-v q8_0 --batch-size 2048 --mlock true

Batch size being from 2k-4k didn't really change results. But I'm not going to argue with a free percent and a half of performance.
1
u/fulgencio_batista avatar
fulgencio_batista
•
28d ago
• Edited 28d ago
emoji:Discord:
Profile Badge for the Achievement Top 1% Poster Top 1% Poster

Hey this is pretty handy! I saw around a 50% boost in tg from my baseline command from the auto-detected command, though I didn't have any luck with my LLM tuning it (no change).

I was trying to run qwen3.5-122b-a10b-reap-40 (~46gb) with 32gb vram.
1
Wide_Veterinarian100
•
28d ago

My noob ass just learned to do this manually, thank you for this!
1
Wise-Hunt7815
•
28d ago

u/raketenkater avatar

fragment_me
•
28d ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Anything serious needs to do benchmarks post changes of perplexity and KLD.
1
u/Leather_Flan5071 avatar
Leather_Flan5071
•
27d ago

this is some nice concept I'm gonna watch for improvements
1
Queasy_Asparagus69
•
28d ago

For a v3 add speculative decoding
0
u/raketenkater avatar
raketenkater
OP •
28d ago

yeah wrote that up already would be another big performance gaining path
1
Queasy_Asparagus69
•
28d ago

And jinja chat templates too ;)
1
u/Professional_Let8686 avatar
Professional_Let8686
•
28d ago

I am using RTX 5070 (12G VRAM) with 128G RAM. What is the best inference tok/s I can expect with these large models?. I am currently running Qwen 3.5 9B unsloth Q4 quant model with q4_1 kv cache and getting around 90 tok/s.
0
u/mrtrly avatar
mrtrly
•
28d ago

The self-referential loop is the clever part here. Most people hand-tune tensor splits once and forget about it, but flag interactions are combinatorial enough that automated search beats human intuition past two GPUs. Quant level probably shifts the optimal split enough that each one needs its own tuning pass.
0
u/dtdisapointingresult avatar
dtdisapointingresult
•
27d ago
emoji:Discord:
Profile Badge for the Achievement Top 1% Poster Top 1% Poster

I wouldn't use a wrapper to launch llama.cpp, especially since I already know it, but I guess it might be useful for complete novices.

    Why is --ai-tune only 8 rounds? What if that's not enough to determine the best flags? Why not run give an option so it pauses after 4 or 8 rounds, proposes results to the user, and they have the option to continue?

    Why are you using the LLM you're running to do the tuning? What if the person is running a dumbed down LLM? You should allow the user to specify an OpenAI-compatible API where the advisor LLM sits

    I certainly hope your tool isn't actually reducing KV cache type to q8 just because the user ran 'llm-server model.gguf.' There should be a flag to never sacrifice cache type. Some people care a lot more about accuracy more than a few more thousand tokens in context size.

IMO your tool could have a lasting utility as a one-shot calibration tool whose final output is the llama-server command which got the best results. Then your tool isn't used again (until the next model).
0
denoflore_ai_guy
•
28d ago

OmG iTs SeLf ImPrOvInG ai!?!?!?! 🤪 but srsly nice stuff.
-2
Clean_Initial_9618
•
28d ago

Say the same thing can I tell claude code to do like give it access to a shell and ask it to run llama-server query it and see the stats and find the best settings and give it access to llamacpp docs. Sorry just asking as I have been trying to find the right flags for my setup as well. Rtx3090 and 64GB system RAM. Trying to run Hermes agent with either gemma4-26B-A4B-it or qwen3.5-27b. any Any help or suggestions would be great. Thank you
-2
u/raketenkater avatar
raketenkater
OP •
28d ago

yes you could do that using claude code as well i think but you would burn your tokens and you need to redo it for diffrent models everytime
5
Clean_Initial_9618
•
28d ago

Makes sense how does the ai-tune work in the background is it safe ? Can I just add that feature to my existing llama-server or do I need to like clone and make llama-server again ??
0
u/raketenkater avatar
raketenkater
OP •
28d ago

so llm-server just build on top of any llama-server binary and for ai-tune being safe it is just your hosted ai reading the -help pages of your binary and based on that tuning the flags of the model currently ran
1
Craftkorb
•
28d ago

If it fits into VRAM go with vllm, much faster
0
