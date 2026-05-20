
nikhilprasanth
•
7h ago

Incredible work. The fact that KV q8_0 is essentially a free lunch even under PPL scrutiny is going to save a lot of VRAM. It’s also interesting to see MXFP4 struggle with speed despite the Unsloth recommendation.
27
simracerman
•
3h ago

Yeah, but the tested context being at 512 tokens is unrealistic. This model is touted as a good coder, and your typical coding agent dumps 10k tokens to start with, you’re gonna find that “free” claim vanish quickly.
10
u/Single_Ring4886 avatar
Single_Ring4886
•
6h ago

Thats what I call thorough testing :)
4
danielhanchen
•
6h ago
• Edited 29m ago
emoji:Discord:
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Awesome work! We're actually going to post our results soon in a few hours hopefully - we just did! https://www.reddit.com/r/LocalLLaMA/comments/1rgel19/new_qwen3535ba3b_unsloth_dynamic_ggufs_benchmarks/ - for those interested we tried over 120 different variants and all are posted here: https://huggingface.co/unsloth/Qwen3.5-35B-A3B-Experiments-GGUF
40
u/gaztrab avatar
gaztrab
OP •
6h ago
emoji:Discord:

Thanks so much Daniel!
9
danielhanchen
•
42m ago
emoji:Discord:
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

As an update - https://www.reddit.com/r/LocalLLaMA/comments/1rgel19/new_qwen3535ba3b_unsloth_dynamic_ggufs_benchmarks/ :)
1
u/IrisColt avatar
IrisColt
•
1h ago

I kneel...
1
u/Live-Crab3086 avatar
Live-Crab3086
•
6h ago

very helpful, thorough analysis. thank you!

anyone willing to speculate if the UD-Q4_K_XL vs Q4_K_M results carry over to UD-Q5_K_XL vs Q5_K_M?
11
u/gaztrab avatar
gaztrab
OP •
6h ago
emoji:Discord:

Likely yes. danielhanchen (Unsloth creator) confirmed the issue is with how UD dynamic quantization handles MoE expert layers in general — it's not specific to the Q4 tier. The standard quant scheme preserves expert structure better. So Q5_K_M should be safer than UD-Q5_K_XL for MoE models, same pattern as Q4.
9
xrvz
•
6h ago
• Edited 3h ago

"how UD dynamic quantization handles MoE expert layers in general" – for all models?

I personally only use MoE at this point for serious work, and that'd be a death sentence for UD; they'd have to do some necromancy to undo that.

Edited: words, not meaning.
3
u/gaztrab avatar
gaztrab
OP •
4h ago
emoji:Discord:

Daniel's comments were specifically about Qwen3.5-35B-A3B — I wouldn't generalize to all MoE models without testing. Different MoE architectures could respond differently to UD quantization. It's plausible the issue is general since UD's dynamic bit allocation may not account for expert layer structure well, but that's speculation, not data. Would need to run KLD on UD vs standard quants for other MoE models to know for sure.
2
u/Maxxim69 avatar
Maxxim69
•
2h ago
• Edited 2h ago

    "MoE expert layers in general" – for all models?

AFAIK, the “new formula” for the UD MOE quants was a recent experiment from ~10 days ago. If you want to check whether a particular quant was affected (i.e. had an unusually large number of its weights in MXFP4), go to its properties and scroll down to the Tensors table.

    that'd be a death sentence for UD; they'd have to do some necromancy to undo that

I would refrain from using sensationalist language. It was just a failed experiment that was promptly noticed by the community and reported to the quant creators who handled the issue with utmost responsibility. Just another day in science and engineering, no need to give it the YouTube thumbnail treatment.
2
Constant-Simple-1234
•
6h ago

So the recommendation is: Unsloth's UD for dense models, but regular Q4_K_M for MoE ?
1
u/gaztrab avatar
gaztrab
OP •
4h ago
emoji:Discord:

That's a reasonable rule of thumb based on what we know so far, but it's only been tested on this one model. UD quants are well-validated on dense models. For MoE, standard quants (Q4_K_M, Q4_K_L etc.) are the safer choice until someone runs similar KLD comparisons on other MoE architectures.
2
u/Lucis_unbra avatar
Lucis_unbra
•
2h ago

I would note, that even the dense models are not "normal". they are still hyrid models. Part transformer part state space (the gated delta net part). the 27B model is also "unusual" with new tech that might follow different "rules"
1
u/Syncfusion avatar u/Syncfusion
•
Promoted
Want to build rich web applications quickly and easily? With 145+ Syncfusion React Components, you can do just that! Plus, our unique document processing libraries make it easy to manipulate Excel, Word, PDF, and PowerPoint files. Get started today with our free 30-day trial!
Learn More
syncfusion.com
Thumbnail image: Want to build rich web applications quickly and easily? With 145+ Syncfusion React Components, you can do just that! Plus, our unique document processing libraries make it easy to manipulate Excel, Word, PDF, and PowerPoint files. Get started today with our free 30-day trial!
u/Ancient_Routine8576 avatar
Ancient_Routine8576
•
6h ago

The data on KV q8_0 being effectively free in terms of perplexity loss is a huge relief for anyone trying to squeeze maximum performance out of a 16GB buffer. It is interesting to see that the instant accuracy drops some users reported are not reflecting in the PPL metrics as that suggests those degradations might be very task specific. Thanks for running these follow up experiments because this level of granular detail is exactly what makes the local LLM community so valuable. I am definitely bookmarking this matrix for my next fine tuning project.
9
u/Front_Eagle739 avatar
Front_Eagle739
•
6h ago

ime q8 kv is a non issue till you have huge contexts and then somehow it falls apart faster than the full 16 bit ones. Seems to exacerbate that cliff where the model starts forgetting things that happened 40-100k tokens ago. At least on glm 4.6 where I did my testing with it
7
u/gaztrab avatar
gaztrab
OP •
5h ago
emoji:Discord:

Important nuance — my E1 tests used 512 token context windows, so I can't speak to 40-100k behavior. It's plausible that quantization errors accumulate over very long sequences in a way that short-context PPL doesn't capture. If you're running huge contexts regularly, that's worth being cautious about. I'll add a caveat to the post.
6
u/Digger412 avatar
Digger412
•
2h ago

I think that's worth more than a caveat, honestly. Measuring on 512 ctx is very small and I'd argue that you'd want to test on a sweep of contexts like with a Needle on a Haystack bench. I've noticed the same thing that quantized KV cache barely impacts KLD at 512 ctx and my conclusion wasn't that it's a free lunch but rather it's either not easily measurable with a KLD test or the default 512 isn't enough to make a measurable difference. Maybe try running KLD with 4k, 8k, 16k, and 32k --ctx-size?

Thanks for this post and running these tests!
3
u/No_Swimming6548 avatar
No_Swimming6548
•
7h ago

Thanks man
20
a_beautiful_rhind
•
6h ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

You can also quant the K and V separate. One of them is responsible for the big hit more than the other. IK_llama has a q_6 and hadamard transforms for K. There's more squeezing if you try.
4
u/gaztrab avatar
gaztrab
OP •
5h ago
emoji:Discord:

Good tip — I quantized K and V together at q8_0 but didn't test them asymmetrically. If one dimension is more sensitive than the other, you could potentially push the tolerant one to q4_0 while keeping the sensitive one at q8_0. More VRAM savings without the quality hit. Something to test in a future round.
3
a_beautiful_rhind
•
4h ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Yes the note is in the original PR for quantized cache on which one is hurt more. I think it's the K but I guess you can also empirically verify it. You might have to compile l.cpp with fa_all_quants
3
u/gaztrab avatar
gaztrab
OP •
4h ago
emoji:Discord:

Time for another round of test then haha
2
u/vast_ai avatar u/vast_ai
•
Promoted
Spin up high-end GPUs for ML training. From $0.25/hr. Deploy in seconds.

    Spin up high-end GPUs for ML training. From $0.25/hr. Deploy in seconds.
    Spin up high-end GPUs for ML training. From $0.25/hr. Deploy in seconds.
    Spin up high-end GPUs for ML training. From $0.25/hr. Deploy in seconds.
    Spin up high-end GPUs for ML training. From $0.25/hr. Deploy in seconds.
    Spin up high-end GPUs for ML training. From $0.25/hr. Deploy in seconds.

cloud.vast.ai
Sign Up
u/_-_David avatar
_-_David
•
5h ago

I think it's a clear mistake to claim that the 27b dense model is "worse quality" based on 2% higher ppl. You might say it degrades more quickly, perhaps. But in benchmarks the 27b absolutely dominates the 35b. I get that this post is from the perspective of "If you have a 16gb GPU, this is what you should choose" but you could either make that more explicitly clear in similar future posts, or not lean so heavily on disparaging the 27b.

With that said, I applaud your diligence and assistance to the community. This was a very well put together post and I appreciate it. I went to download bartowski's Q4_K_L model instantly on your recommendation, and I'll be eating my free KV lunch at q8 thanks to you. It just felt a bit odd to see my new favorite model, the 27b dense that I'm running fully in VRAM, tossed to the side and spat upon. Which again, is totally fair if we're talking a 5080 User's Guide! If the title of the post had been that, I think I wouldn't have noticed.
4
u/gaztrab avatar
gaztrab
OP •
5h ago
emoji:Discord:

"Worse quality" was sloppy framing on my part. PPL on WikiText-2 measures one narrow thing; the 27B dominates on actual capability benchmarks and instruction following. What I should have said is: on a 16GB GPU where the 27B runs at 7 tok/s vs 75 tok/s for the MoE, the speed difference makes it impractical for interactive use. But if you can fit 27B fully in VRAM (4090, 5090), it's arguably the better model. I'll update the wording. Thanks!
3
theghost3172
•
5h ago
• Edited 5h ago

"The 35B-A3B MoE dominates on both speed AND quality"

that is not true. you cant compare different llms with perplexity. different llms have different distributions so they will have different perplexity irrespective of quality. and Moe will always have lower quality than dense. but ofc its much faster.

but overall excellent work
6
u/gaztrab avatar
gaztrab
OP •
5h ago
emoji:Discord:

Yeah you're right, what I should have said was "MoE dominates on both speed and quality for 16GB VRAM". Thanks anyway!
3
u/Corosus avatar
Corosus
•
6h ago

absolutely amazing insight tysm, gonna use fit that way and try that quant
3
kaeptnphlop
•
6h ago

Very insightful! Thank you for testing this out. That’s a lot of work!
3
Pawderr
•
6h ago

Can someone please explain what this means? I just started with local llms 
3
u/gaztrab avatar
gaztrab
OP •
6h ago
emoji:Discord:

I'm testing the best way to run a large AI model (Qwen3.5-35B-A3B) on a single gaming GPU (RTX 5080, 16GB). The model is too big to fit entirely in the GPU, so parts of it run on the CPU — finding the optimal split is what most of these experiments are about.

If you're just getting started, the takeaway is:

    Model: Qwen3.5-35B-A3B with Q4_K_M quantization (a way to compress the model so it fits)

    Engine: https://github.com/ggml-org/llama.cpp — free, open source, runs on any NVIDIA GPU

    Key settings: --fit on, -ctk q8_0 -ctv q8_0, -fa on, and do NOT add -b 4096 -ub 4096

This gets ~75 tokens/second, which is faster than most people read.
29
uxl
•
6h ago

As a guy with a 5080 mobile and 64GB ram, thanks! Is llamcpp better than kobold?
5
u/gaztrab avatar
gaztrab
OP •
6h ago
emoji:Discord:

KoboldCpp is built on the same llama.cpp backend, so raw inference speed should be nearly identical. The main differences: KoboldCpp bundles a web UI aimed at creative writing/RP, while llama-server gives you a bare OpenAI-compatible API. The reason I'd recommend mainline llama.cpp is it picks up new features faster — --fit on (which gave us the biggest speed gain here) landed in mainline weeks before forks typically adopt it.

Your 5080 mobile + 64GB RAM should work great with the same config. One heads up: mobile 5080 has lower memory bandwidth than desktop, so expect somewhat lower tok/s, but the optimal settings are the same.
9
uxl
•
5h ago

Do you have Patreon or some means by which I can buy you a coffee? Greatly appreciate your work and help.
4
u/gaztrab avatar
gaztrab
OP •
5h ago
emoji:Discord:

Oh man, thank you very much for your kindness, but there's no need. I grew up on the early internet and believe that knowledge should be shared for free.
13
mrdevlar
•
1h ago

Doing the machine gods work dude, keep it up. May you have an excellent day.
2
u/gaztrab avatar
gaztrab
OP •
1m ago
emoji:Discord:

You too, my friend.
1
Pawderr
•
5h ago

These settings are only for text i suppose? Would it work for video related tasks?
1
u/gaztrab avatar
gaztrab
OP •
5h ago
emoji:Discord:

I havent tried video yet, but if you ever do please share!
2
hsoj95
•
4h ago
Llama 8B

Based on all this, I guess it's time for me to switch from Ollama to Llama.cpp? Any advice for someone switching, been using the Docker for Ollama.
1
u/gaztrab avatar
gaztrab
OP •
4h ago
emoji:Discord:

If you're comfortable with Docker then you'll start easily since there are many images out there. You could try my repo first, it was made to act as a local server too!
2
LostDrengr
•
3h ago

I have been embarking on the same also having a 5080. I have used docker desktop to offer vLLM and using the Tensorrt option, with the latter exploring the NVFP4 for increased speed. I also use LM Studio but have forgot about the ollama route.

With these new models being huge its trying to get the quantised flavour that can run well, so this is really helpful appreciate all the 16GB coverage!
1
u/prescorn avatar
prescorn
•
6h ago

Nice work! This will be useful for some of my 96GB experiments on the weekend.
3
u/ArckToons avatar
ArckToons
•
5h ago

Great tests with a lot of useful conclusions. I disagree with “The 27B dense is only worth considering if you need a non-MoE model for compatibility reasons.” I don’t think it’s only about compatibility, but about use cases.

If you need speed, 35B is the right call. But if you want more quality (even though in most use cases the quality is similar), better instruction-following, and more predictable behavior, 27B seems like the better choice.

In my case, I have an RTX 4090 and I run it with OpenCode. I tested both 27B Q4_KM and 35B Q4_KM, and the 27B did better with my orchestrator/sub-agent setup. I’m not saying 27B is objectively superior—this depends on the use case and whether slower inference is acceptable—but I don’t think the decision comes down to compatibility.

One question: does KV quantization affect KL? Would it be worth running a test, or not?
3
u/gaztrab avatar
gaztrab
OP •
5h ago
emoji:Discord:

Agreed with your take on 27B — see my reply to _-_David above, same point applies. On a 4090 with full VRAM headroom, 27B is a totally valid choice, especially for agentic workflows where quality matters more than tok/s.

On your KV quant + KLD question: I tested KV quant impact on PPL (E1) but didn't run KLD specifically across KV quant levels. PPL showed < 0.4% difference between f16/q8_0/q4_0, so I'd expect KLD to be similarly minimal — but that's an assumption, not data. Worth testing if someone wants to be thorough.
3
u/marcoc2 avatar
marcoc2
•
6h ago

Does anyone have config or link for a 4090-24gb?
2
u/gaztrab avatar
gaztrab
OP •
6h ago
emoji:Discord:

Same flags, just let --fit do the work:

./llama-server \
-m ./Qwen3.5-35B-A3B-Q4_K_M.gguf \
-c 65536 \
--fit on \
-fa on \
-t 20 \
--no-mmap \
--jinja \
-ctk q8_0 \
-ctv q8_0

With 24GB VRAM, --fit should keep nearly all of Q4_K_M (~20 GB) on GPU — you'll likely see higher tok/s than my 16GB results. Tune -t for your CPU (20 is optimal for my 9950X, try physical_cores × 0.6 as a starting point and sweep from there). You could even try Q8_0 (36.9 GB) with partial offload — at 24GB you'd get significantly more layers on GPU than I can.
6
u/marcoc2 avatar
marcoc2
•
5h ago

Thank you 🙏
2
Life-Screen-9923
•
6h ago

Great job, thank you! 🔥🔥🔥
2
maxpayne07
•
6h ago

Kudos!!
2
JoseGemez
•
6h ago

This weekend i try on a 5060 ti 16gb! Many thanks
2
MaCl0wSt
•
6h ago

wow fantastic post, thanks
2
ayylmaonade
•
6h ago
emoji:Discord:

Thank you so much for the MXFP4 testing! Happy to see that quantizing the KV cache doesn't impact performance too. Really appreciate all the effort. :)
2
joshbates15
•
6h ago

This is amazing work! Thank you for sharing.
2
savenx
•
6h ago

Thanks for the tests, very helpful! I have a question: Im using a RX6900XT 16GB vram and i have 32GB ram, which version should i use? I tried Q4 on LM studio and its pretty fast, but when i try to use it on OpenCode (agentic use) it becomes unusable
2
u/gaztrab avatar
gaztrab
OP •
5h ago
emoji:Discord:

All my tests were NVIDIA/CUDA, so I can't give specific AMD numbers — ROCm/Vulkan backends may behave differently. For the RX 6900 XT with 16GB VRAM, Q4_K_M with --fit on should still be the right choice. The OpenCode issue is probably separate — likely related to context length or thinking mode (Qwen3.5 has thinking enabled by default which consumes extra tokens). Try disable it, or check if OpenCode is sending very long system prompts that fill your context.
1
u/wisepal_app avatar
wisepal_app
•
5h ago

This is the best explanation on this sub i saw, about a technical topic. very informative and simple. thank you for your hard work.
2
cookieGaboo24
•
5h ago

Great test, nice Work and thank you. One question, how did you guys get those 50t/s on 8gb VRAM? I did the same offloading on my 3060 12gb and only get around 30t/s. Did you just offload them all on the 5080 or used a different card?
2
u/gaztrab avatar
gaztrab
OP •
5h ago
emoji:Discord:

Right, sorry about this, I ran the test on my 5080. I will update the post to clarify this!
4
cookieGaboo24
•
5h ago

Thanks, figured as much but better ask again. Not mentioning the gap in GPU generation, ram generation (ddr4 vs ddr5) and the absurdly stronger CPU (R5 3600 here) which I already partially considered to be at fault, I still consider my 30t/s a win.
2
u/Technical-Earth-3254 avatar
Technical-Earth-3254
•
5h ago
llama.cpp

Goated post, thank you for all the effort you did put into this
2
allattention
•
5h ago
• Edited 5h ago

Awesome work, much appreciated! I thought we used -u and -ub to make reading large context after a KV reset (which happens often if you use opencode) faster. I’ll try without them now.
2
u/gaztrab avatar
gaztrab
OP •
5h ago
emoji:Discord:

Good point about KV resets. I will also test them in future round. Thanks!
3
Old-Sherbert-4495
•
5h ago

i didn't understand 90% of this, I was trying my fullest to get 27b q4 working faster in my 16vram and 32 ram setup. when i have fit on, it leaves a lot of vram and cpu is 100% (i did quantize cache q8.) moe 35b was definitely faster. but that also leaves a few gig vram and the cpu goes bananas. how can i get the best of the available vram any advice
2
u/gaztrab avatar
gaztrab
OP •
5h ago
emoji:Discord:

A few things to check:

    Make sure you're NOT using -b 4096 -ub 4096 — those eat VRAM that --fit needs

    Add --no-mmap — loads the full model into RAM upfront, gives --fit a clearer picture of available memory

    With 32GB RAM you're tight — try reducing context to -c 32768 instead of 65536, which frees KV cache VRAM for more expert layers on GPU

    CPU at 100% is normal and expected — that's the CPU computing the expert layers that don't fit on GPU. The goal is to minimize how many go to CPU, not eliminate CPU usage entirely.

Full command:

./llama-server -m ./Qwen3.5-35B-A3B-Q4_K_M.gguf -c 32768 --fit on -fa on -t 20 --no-mmap --jinja -ctk q8_0 -ctv q8_0

Tune -t for your CPU — try physical cores × 0.6 as a starting point.
3
Old-Sherbert-4495
•
4h ago

thnx a lot will give this a shot
2
Old-Sherbert-4495
•
4h ago

one more thing, do i have hope for 27b or should i just forget
2
u/gaztrab avatar
gaztrab
OP •
4h ago
emoji:Discord:

I think it's depends on how patience you are, I would advise testing both and see if the quality + speed is comfortable for you
2
u/Corosus avatar
Corosus
•
5h ago
• Edited 5h ago

Some quick testing, using --fit for me tanks performance, -ngl 999 --n-cpu-moe 24 works best on my pc, 5070 ti (other gpus disabled), 128gb ddr4 3200mhz. Maybe because I'm still using vulkan.

I guess this goes to show theres no universal solution, gotta find out what works best for your hardware:

llama-b8173-bin-win-vulkan-x64\llama-server --model ./e/Qwen3.5-35B-A3B-Q4_K_M.gguf --host 0.0.0.0 --port 8080 -ctk q8_0 -ctv q8_0 -ngl 999 --n-cpu-moe 24 --flash-attn on --jinja -c 48000 -t 20

-ngl 999 --n-cpu-moe 24

33tps
	
llama_memory_breakdown_print: | memory breakdown [MiB]    | total   free     self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - Vulkan0 (RTX 5070 Ti) | 15907 = 4641 + (10162 =  8845 +     750 +     566) +        1103 |
llama_memory_breakdown_print: |   - Host                  |                 12033 = 11931 +       0 +     102                |
	


	
llama-b8173-bin-win-vulkan-x64\llama-server --model ./e/Qwen3.5-35B-A3B-Q4_K_M.gguf --host 0.0.0.0 --port 8080 -ctk q8_0 -ctv q8_0 --fit on --flash-attn on --jinja -c 48000 -t 20

--fit on

13 tps
	
llama_memory_breakdown_print: | memory breakdown [MiB]    | total   free     self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - Vulkan0 (RTX 5070 Ti) | 15907 =  890 + (13825 = 12574 +     750 +     501) +        1190 |
llama_memory_breakdown_print: |   - Host                  |                 19916 = 19814 +       0 +     102                |
	



llama-b8173-bin-win-vulkan-x64\llama-server --model ./e/Qwen3.5-35B-A3B-Q4_K_M.gguf --host 0.0.0.0 --port 8080 -ctk q8_0 -ctv q8_0 --fit on -ot "exps=CPU" --flash-attn on --jinja -c 48000 -t 20

--fit on -ot "exps=CPU"

24tps

llama_memory_breakdown_print: | memory breakdown [MiB]    | total    free     self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - Vulkan0 (RTX 5070 Ti) | 15907 = 12152 + ( 2656 =  1339 +     750 +     566) +        1098 |
llama_memory_breakdown_print: |   - Host                  |                  19916 = 19814 +       0 +     102                |

I also reran the --fit on test with b8149, same slow result

edit: realized i forgot --no-mmap to go with --fit on, prompt intake is still insanely slow so tps is likely also slow
2
u/gaztrab avatar
gaztrab
OP •
5h ago
emoji:Discord:

Very interesting. I will note your result down, thanks for sharing!
2
u/gaztrab avatar
gaztrab
OP •
5h ago
emoji:Discord:

Looking at your memory breakdowns, --fit on is allocating 12574 MiB model to GPU vs 8845 MiB with manual --n-cpu-moe 24 — but that extra GPU allocation makes it slower (13 vs 33 tok/s). That strongly suggests --fit isn't optimizing well for Vulkan — it's probably tuned for CUDA compute characteristics and over-loading the GPU with layers that would run faster on CPU via your DDR4. Your manual --n-cpu-moe 24 is the right call for Vulkan setups. I'll add a note to the post that --fit results are CUDA-specific and Vulkan users should stick with manual offloading.
2
u/Danmoreng avatar
Danmoreng
•
3h ago

I believe with —fit you should also use —fit-ctx instead of just —c. Also, if you want to use the vision capability of the model, you have to either put the vision model on CPU or use —fit-target 1536 to leave space for the vision part on GPU. I am running on very similar settings on my notebook with a 5080 mobile and can confirm initially having 74 t/s, for longer context it then falls of to around 66 t/s.

My server configuration can be found here: https://github.com/Danmoreng/local-qwen3-coder-env?tab=readme-ov-file#server-optimization-details
2
u/gaztrab avatar
gaztrab
OP •
3h ago
emoji:Discord:

Interesting, I haven't tested --fit-ctx, only --fit on with -c. And the --fit-target 1536 tip for vision is great, I have the mmproj downloaded but haven't smoke-tested it yet. Your config repo is really useful. I will properly test vision on the next round of experiment!
1
u/catlilface69 avatar
catlilface69
•
6h ago

There are doubts about your experiments. What do you mean q4 quant with q4 kv cache is more accurate?
3
u/gaztrab avatar
gaztrab
OP •
6h ago
emoji:Discord:

Hollup, let me verify this again, back in a min
6
u/gaztrab avatar
gaztrab
OP •
6h ago
emoji:Discord:

I reran the full E1 experiment just now and got identical numbers both times. PPL evaluation is deterministic on the same dataset, so the sub-0.4% differences are real and reproducible, just too small to matter in practice.

The slight "improvement" with q8_0 KV is likely a minor rounding effect from quantization — essentially noise at that scale. The takeaway is that KV q8_0 doesn't hurt quality at all, so the throughput gain is free.

You can reproduce it yourself from https://github.com/gaztrabisme/llm-server

./scripts/run-experiment.sh e1

Runs all 6 PPL evaluations (~25 min) and prints the comparison table at the end.
6
bigvenn
•
5h ago

Incredible work man, this is science
1
pmttyji
•
5h ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Big thanks for this thread. Appreciate your time & your experiments.

Could you please add one more stuff(on Experiment 1) in this thread?

Experiment with -ctk q8_0 -ctv q4_0 because K is sensitive while V isn't. I remember few people do use this combination instead q4 on both.

Thanks.
1
u/gaztrab avatar
gaztrab
OP •
4h ago
emoji:Discord:

Good suggestion — u/a_beautiful_rhind mentioned the same thing about K being more sensitive than V. I tested K and V together (both q8_0, both q4_0) but didn't test the asymmetric combo. Adding it to my list for the next round of experiments.
3
RMK137
•
4h ago

Great read, thank you for putting this out.
1
KierkegaardsSisyphus
•
4h ago

I'm not understanding your speeds for MXFP4. I have a 5080 and I get about 77 tk/s on short contexts. I use fit target, otherwise image processing goes OOM.

exec "$BINARY" \

-m "$MODEL_PATH" \

--mmproj "$MMPROJ_PATH" \

-c 65536 \

-fa on \

--fit on \

--port 5001 \

--fit-target 1500 \

-ctk q8_0 \

-ctv q8_0 \

--jinja \

--no-mmap

Side note: I prefer running with -b 4096 and -ub 2048. For me, the massively improved processing speed is worth losing a few tokens of text gen speed.
1
u/gaztrab avatar
gaztrab
OP •
4h ago
emoji:Discord:

I will properly test MXFP4 on the next round with your config as reference. Thanks my dude!
3
u/Psyko38 avatar
Psyko38
•
4h ago

So, this will work on a 16GB GPU of VRAM only, without the need for system RAM.
0
u/gaztrab avatar
gaztrab
OP •
4h ago
emoji:Discord:

No — the model is ~20 GB at Q4_K_M, so it won't fit entirely in 16GB VRAM. You need system RAM for the expert layers that overflow to CPU. With 32GB RAM you'll be fine but tight (try -c 32768 instead of 65536). With 64GB+ RAM you have plenty of headroom. The more RAM bandwidth you have (DDR5 > DDR4), the faster the CPU-side expert computation will be.
2
u/Psyko38 avatar
Psyko38
•
4h ago

Okay, because currently my 9060xt 16gb and my Ryzen 5500 on 32gb RAM in DDR4 3400 allowed me to reach 36 tok/s with Unsloth's Q3KXL. So, with your optimizations, maybe 40 tok/s.
1
u/gaztrab avatar
gaztrab
OP •
4h ago
emoji:Discord:

If you get better speed, please share with us!
1
u/Psyko38 avatar
Psyko38
•
4h ago

I would, but right now, I'm not home.
0
u/Psyko38 avatar
Psyko38
•
1h ago

Benchmark Report: Llama.cpp (ROCm) on AMD

1. Hardware and Software Configuration

Hardware:

* CPU: AMD Ryzen 5 5500 (6 Cores / 12 Threads)

* RAM: 32 GB DDR4 @ 3400 MHz

* GPU: AMD Rx 9060 XT 16GB

* OS Win 11 25h2

* llama.cpp llama-b8145-bin-win-hip-radeon-x64

Software:

* Backend: ROCm (via llama.cpp)

* Model: Qwen3.5-35B-A3B (Version Q3_K_XL)

* Type: MoE Architecture Mixture of Experts

* Quantization: UD-Q3_K_XL (Q3 quantization)

* Tool: llama-server

2. Benchmark Commands

Test 1:

llama-server -m "...\Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf" -c 16384 -ngl 999 -fa on -t 16 -b 4096 -ub 4096 --jinja --no-mmap -ot "blk\.([0-9]|[1-2][0-9]|30)\.=ROCm0,exps=CPU" -ctk q8_0 -ctv q8_0

Test 2:

llama-server -m "...\Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf" -c 16384 -ngl 999 -fa on -t 16 -b 4096 -ub 4096 --jinja --no-mmap -ot "blk\.([0-9]|[1-2][0-9]|30)\.=ROCm0,exps=CPU

Test 3:

llama-server -m "...\Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf" -c 16384 --fit on -fa on -t 16 --no-mmap --jinja -ctk q8_0 -ctv q8_0

3. Results Table
Test 	Context 	KV Cache 	GPU Offload 	Threads 	Performance (t/s)
1 	16,384 	Q8_0 	30+ layers 	16 	37.74
2 	16,384 	Full 	30+ layers 	16 	38.26
3 	16,384 	Q8_0 	Auto / Fit 	16 	15.82
1
u/mdziekon avatar
mdziekon
•
4h ago

Regarding the "Experiment 4: --fit Tuning", could you test how does not using -b 4096 affect prompt processing speeds? Token generation is one thing, but prompt processing, especially for coding, is even more important for session start (which is especially painful for orchestrating agents when PP is slow).
0
u/gaztrab avatar
gaztrab
OP •
4h ago
emoji:Discord:

I will test this on the next round and tag you in. Thanks for the suggestion!
2
u/mdziekon avatar
mdziekon
•
3h ago

Awesome! Great work BTW :)
1
u/DepravedPrecedence avatar
DepravedPrecedence
•
6h ago

Is it possible to use these flags in LM Studio? I think it doesn't allow setting flags of llama.cpp like that?
1
u/gaztrab avatar
gaztrab
OP •
6h ago
emoji:Discord:

LM Studio does expose some of these — context length, flash attention, GPU layers, and KV cache quantization (check for llamaKCacheQuantizationType / llamaVCacheQuantizationType in load settings, requires flash attention enabled). However, --fit on (the auto VRAM management that gave us the biggest speed gain) is not available in LM Studio — it's a recent llama.cpp feature.

If you want the full config, you'll need to run llama-server directly. The Docker setup in https://github.com/gaztrabisme/llm-server makes it straightforward, or you can just build llama.cpp from source and run the command from my post.
3
AvidCyclist250
•
6h ago

I suppose you could half-ass the same gains by manually fine-tuning context length?
1
ilintar
•
6h ago
emoji:Discord:
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Very nice benchmark, I hope it really puts to rest a few stupid myths, including "KV cache quantization absolutely kills quality for coding" and "MXFP4 is the best 4-bit quant ever".
1
u/gaztrab avatar
gaztrab
OP •
6h ago
emoji:Discord:

I wouldnt say my experiments killed the myths, since comparing to the numbers of models and quants out there, this is just a small drop in the ocean. But thanks!
1
ilintar
•
5h ago
emoji:Discord:
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

It takes just one counterexample to kill a general statement, that's the beauty of it 😀
2
u/R_Duncan avatar
R_Duncan
•
6h ago

Sorry, or your test were shallow or there is a mistake or is a lie, but MXFP4_MOE " 34-42% slower " than Q4_K_M is not true. Anyone can verify. (4060 laptop with CUDA backend here)

Given the same question to both models, I got no noticeable slowdown of MXFP4_MOE.
1
u/gaztrab avatar
gaztrab
OP •
6h ago
emoji:Discord:

Verified — I just re-checked raw benchmark logs and the numbers hold. Both quants tested with identical configs (--fit on, same threads, same KV q8_0, same Docker image). Raw data is public: https://github.com/gaztrabisme/llm-server under benchmarks/.

That said — my results are specific to RTX 5080 16GB. The relative gap could be different on your 4060 laptop depending on how much overflows to CPU and how MXFP4 dequant performs there. What tok/s are you seeing for each on your setup?
3
Constant-Simple-1234
•
6h ago

Yes. Different architectures may influence this. I run on Vulkan on iGPU - much smaller differences between Q4_K_M and MXFP4. But I will confirm on 5060 TI.
1
u/gaztrab avatar
gaztrab
OP •
6h ago
emoji:Discord:

Hey bro. Let me double check my method and rerun MXFP4 experiment to be sure.
1
u/DigiDecode_ avatar
DigiDecode_
•
6h ago

from what i know is that MXFP4 is designed & optimised for RTX 5000 series and future
Q4_K_M is more suited for RTX 4000 & 3000 series
1
u/MrQ_dos40 avatar
MrQ_dos40
•
6h ago

This is a fantastic deep dive into Qwen3.5-35B-A3B performance! I'm particularly interested in the --fit on results. Have you considered testing with different batch sizes to see if that impacts the token/s further, especially with the 16GB VRAM constraint?
1
u/gaztrab avatar
gaztrab
OP •
5h ago
emoji:Discord:

Yes — that's actually what Experiment 4 was about! The key finding: with --fit on, you should NOT set -b 4096 -ub 4096. Those batch buffers pre-allocate VRAM that --fit then can't use for expert layers on GPU. Removing them entirely gave us 74.7 vs 64.3 tok/s — a 16% improvement just from letting --fit have the VRAM. I also tested --fit-target 256 (smaller batch allocation) but it only partially helped. The simple answer: just don't set -b/-ub at all and let --fit manage everything.
2
u/DHasselhoff77 avatar
DHasselhoff77
•
3h ago

Weren't the custom batch sizes there to speed up prompt processing? So by removing them you are trading off PP speed for generation speed by an unknown amount. Not always a win.

A very clear experiment still. I appreciate the direct writing style and presentation. Thank you!
2
u/soyalemujica avatar
soyalemujica
•
5h ago

Mind you share what ollama command did you use to run the 8Q and 4K_M models for 16gb vram ?
1
u/gaztrab avatar
gaztrab
OP •
5h ago
emoji:Discord:

I actually didn't use Ollama — all tests used llama.cpp server directly. One of my research findings was that Ollama is ~3x slower for MoE models because it doesn't support expert-level offloading (it splits entire transformer layers between GPU/CPU instead of just the expert FFNs). There's an https://github.com/ollama/ollama/pull/12333 to add num_moe_offload but it hasn't merged yet.
1
u/soyalemujica avatar
soyalemujica
•
4h ago

Thank you, mind you share your compiled llama.cpp with that sm_120 you mentioned ? I am having a hard time compiling it
1
u/Lrrrrr avatar
Lrrrrr
•
4h ago

I fuckin love you bro. Got a 5060Ti16gb I did some tests on. Your data is so valuable for us GPU poors 😂 You use q4km from unsloth right?
1
u/gaztrab avatar
gaztrab
OP •
4h ago
emoji:Discord:

I fuckin love u too (no homo). Yeah that's quant I used :p
2
u/soyalemujica avatar
soyalemujica
•
3h ago

mind you share your compiled llama.cpp with that sm_120 you mentioned ? I am having a hard time compiling it for my rtx 5060ti
1
u/gaztrab avatar
gaztrab
OP •
3h ago
emoji:Discord:

The 5060 Ti is also Blackwell (sm_120), so the same build works. Easiest path is using our Dockerfile which handles everything:

git clone https://github.com/gaztrabisme/llm-server
cd llm-server
docker build -f docker/Dockerfile.llama-cpp --build-arg LLAMA_CPP_REF=b8149 -t llm-server/llama-cpp:latest-fit docker/

That builds llama.cpp from source with CUDA 12.8 + sm_120. You need Docker + NVIDIA Container Toolkit installed. If you want to build without Docker, the key CMake flags are: -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_FA_ALL_QUANTS=ON with CUDA 12.8+.
1
u/leonbollerup avatar
leonbollerup
•
3h ago

I have a 3090 and RTX 4000 pro and can run the same tests if you show me what/how you ran them
1
u/gaztrab avatar
gaztrab
OP •
3h ago
emoji:Discord:

Everything's in my repo: https://github.com/gaztrabisme/llm-server (also optimized for coding agent too, just point them to CLAUDE.md)

Quick start:

    Build the Docker image: docker build -f docker/Dockerfile.llama-cpp --build-arg LLAMA_CPP_REF=b8149 -t llm-server/llama-cpp:latest-fit docker/

    Download Q4_K_M: huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q4_K_M.gguf --local-dir ./models

    Run a benchmark: ./scripts/bench.sh llama-cpp s006-e4-fit-nobatch

With a 3090 (24GB) you'll have more VRAM headroom than me — would love to see your numbers.
0
u/Lucis_unbra avatar
Lucis_unbra
•
2h ago

I would note that while the ppl might be fine, it's not free. The token generation speed drops much faster, at least on my rig with windows.

At ~50k with an iq4_xs quant, F16 gives me around 75tps, down from 86.

Q8_0 at that CTX ends up at 65tps. That's a 10tps loss.

If this was not fully on the GPU, we can expect this to get worse.

At Q8_0, I start off at about 42, and this then drops to 39tps.

If I drop the KV cache down to 8 bit again, it drops to 36tps.

Now this is on a decently powerful system with a 3090 and a Ryzen 9 7900x. But depending on the configuration, and the model, this could get much worse. For the 27B dense model that is already hard enough to run? Not fun.
1
u/Dthen_ avatar
Dthen_
•
2h ago

Is there a guide or config for manually offloading on AMD/Vulkan/RoCM?
1
u/soshulmedia avatar
soshulmedia
•
2h ago
• Edited 2h ago

This repo and quantizing team came up recently:

https://huggingface.co/AesSedai/Qwen3.5-35B-A3B-GGUF

Did you do a comparison? (If not, can you?) They have some quants (for other qwen3.5 sizes) that compared favorably to unsloth's.

EDIT: Oh and thank you of course for doing all these tests!
1
Hacket1967
•
1h ago

Impresionante trabajo ,felicidades ¿Que compilación usastes, la de unsloth? ¿Has probado está :https://huggingface.co/AesSedai/Qwen3.5-35B-A3B-GGUF?
1
u/IrisColt avatar
IrisColt
•
57m ago

THANKS!!!
1
u/mintybadgerme avatar
mintybadgerme
•
52m ago

Sorry for a boring question but...

I don't suppose you have any settings for a RTX 5060ti 16GB VRAM with 64GB RAM Intel?

That would be very helpful as I'm trying to work out how to use the model as a coding tool. Thanks. :)
1
u/jpbarcelos avatar
jpbarcelos
•
38m ago

Hi, I'm just starting my local LLM journey on a Mac mini 16gb (which currently run qwen3-14b).

I've been reading that you have to have 32gb to be able to run qwen3.5, yet you mention 16gb video card.

Can I replicate this on my Mac?

Or am I missing something here?
1
u/Chromix_ avatar
Chromix_
•
41m ago
emoji:Discord:

Thanks for taking the time for the extensive follow-up and immediately making edits taking the further feedback into account. That's refreshing to see.

I randomly came across this, as I didn't get any notification for this, despite being mentioned. It worked in your previous comment. Maybe notifications simply got skipped for your post as you mentioned so many others?

Btw: Without the batch setting your token generation is faster, but prompt processing gets slower (only because you don't have enough VRAM for full offload). Tough choice depending on the use-case.
1
u/Majesticeuphoria avatar
Majesticeuphoria
•
15m ago

This is some serious work, nice!
1
Gringe8
•
5h ago

Your dense vs moe speed part is severly flawed. You do mention it needs to fit fully in vram, but you test one that doesnt fit fully in vram. You also dont mention prompt processing speed. I get 2000t/s pp and 28t/s tg on 27b q8.

I do like your other tests. If you wamt to do more i would love to see quality differences between q4km and iq4ks. Then you could also test the speed since it should fit fully in your 5080.
1
UniversalJS
•
12h ago

Great post and experiments! Inspired by your findings, I went a different direction: instead of optimizing Q4_K_M, I tested whether a smaller quant that fits mostly in VRAM could beat it on speed.

Setup: RTX 5080 16GB, Intel Core Ultra 9 285K, llama.cpp built from source with CUDA 13.1 + native sm_120 (Blackwell), using your recommended flags (no batch flags, --fit on, KV q8_0).

The problem with Q4_K_M on 16GB: The model is ~20 GB, so --fit offloads ~9 GB of expert weights to CPU. GPU sits at ~45% utilization waiting for CPU experts. That's the bottleneck.

The idea: Q2_K_L (bartowski) is only ~13.8 GB. At 128k context, almost all expert weights stay on GPU (~800 MiB on CPU, mostly the embed/output layer from the 248K vocab — unavoidable).

Results: 72% faster than Q4_K_M, with 2x the context. Even at 250k context (near the model's 262k training length), Q2_K_L still does 108 tok/s — 45% faster than Q4_K_M at 65k. The trade-off is obviously quality. Q2_K_L will have noticeably worse perplexity than Q4_K_M. But for interactive use, code generation, and tasks where speed matters more than peak accuracy, it's a compelling option on 16 GB cards.

Interesting finding on context scaling: As context increases, --fit progressively offloads more expert layers to CPU to make room for the KV cache. The 515 MiB always on CPU (embed/output) is fixed, but at 250k context, total CPU offload grows to 2.3 GB. The speed degradation is graceful though — only 16% slower going from 128k to 250k.

Also worth noting: Building from source with CUDA 13.1 matters for RTX 50-series. The prebuilt binaries use CUDA 12.4 which lacks sm_120 — you get JIT-compiled PTX from sm_89 instead of native Blackwell kernels.

Launch command (128k context, sweet spot): ./llama-server -m ./Qwen3.5-35B-A3B-Q2_K_L.gguf -c 131072 --fit on -fa on -t 20 --no-mmap --jinja -ctk q8_0 -ctv q8_0

Would love to see KLD/PPL numbers for Q2_K_L if anyone has the patience to run them. My gut says it's worse than Q4_K_M but the speed advantage is hard to ignore.
1
u/moahmo88 avatar
moahmo88
•
11h ago

I think you should try Qwen3.5-27B-GGUF Q3_K_S or Q3_K_M.
2
u/moahmo88 avatar
moahmo88
•
16h ago

You can try AesSedai/Qwen3.5-35B-A3B-GGUF Q5_K_M. 5070ti works well.Surprise！
1
u/moahmo88 avatar
moahmo88
•
16h ago

Amazing! Thanks.
1
u/soyalemujica avatar
soyalemujica
•
16h ago
• Edited 16h ago

Alright, I gave this tutorial a try, compiled llama.cpp with the params as described, running on RTX5060ti 16GB + 64gb DDR5 6400mts/s, and I'm only getting 50t/s, did I do something wrong? Using CUDA 13.1 and latest NVIDIA drivers.
Edit: Getting 55/s , which is an increase of 10t/s in LM Studio and precompiled public llama libraries, this is nice! The difference of 20 tokens might be because the 5080 has 960gb/s bandwidth vs 460gb/s bandwidth on my 5060TI I suppose...
1
u/soyalemujica avatar
soyalemujica
•
14h ago

Trying now the BF16 MXFP4_MOE model, it's giving me 35t/s and also thinking LESS and giving me a result quicker than the Q4_M model.
1
u/maho_Yun avatar
maho_Yun
•
20h ago

Thanks I have done tested base on this with my 5060ti and CLIP enabled

Diff Quant and Flag tested with mmproj-BF16.
--ctx-size 131072 -n 32768 --flash-attn on --kv-offload --no-mmap -ctk q8_0 -ctv q8_0
Full Tom Sawyer.txt with promt: Write a Essay About it.
Model & Config     Prompt Eval (t/s)     Eval/Gen (t/s)     Total Time (ms)
Unsloth MXFP4 ncpumoe 24 b2024 ub 1024 (CUDA 12.4)     875.56     30.10     202,516
Unsloth MXFP4 ncpumoe 24 b2024 ub 1024     929.55     32.32     186,634
Unsloth-UD Q4_K_M ncpumoe 24 b2024 ub1024     860.97     34.34     183,707
Unsloth-UD Q4_K_M fit on fit-target 1536     813.95     38.91     186,154
Aessedai Q4_K_M ncpumoe30 b2024 ub1024     867.85     30.93     179,110
Aessedai Q4_K_M fit on fit-target 1536     870.69     35.26     178,969
Aessedai Q4_K_M fit on     199.74     25.45     613,891
1
KeldenL
•
1d ago

dude this is incredible. i was doing tests on my end too and got tired at how slow it was (probably should've done it on lower context lengths)

one thing that i may or may not have missed in the post, but who's Q4 quant are you using? unsloths? or others? i remember seeing another post about different quants
1
u/CATLLM avatar
CATLLM
•
1d ago

This is amazing thank you!

Is it worth using

--no-kv-offload

to offload KV cache into ram?

1
bobaburger
•
1d ago
emoji:Discord:

Great work! I was thinking of making a separate post, but since this is also in the 16 GB VRAM category, I'm adding my findings for anyone using 5060 Ti here. My setup also using 32 GB DDR5 RAM.

All tests were done with q8_0 kv cache, context window 128k, pp 18k, tg 768, depth 0. Why? Because this closes to a cold start with Claude Code. You can adjust the context window to lower for a bit more performance gain.
Model     pp18432 (t/s)     tg768 (t/s)     Mean KLD
Unsloth UD-Q4_K_M     1047.84     40.64     0.0192
AesSedai Q4_K_M     928.10     34.89     0.0096
Unsloth IQ3_S     1465.81     44.77     0.0457
Unsloth MXFP4     1186.50     38.32     0.0272
Unsloth UD-Q4_K_XL     1002.84     36.59     0.0137

Mean KLD was from Unsloth's data.

AesSedai's Q4_K_M has the best mean KLD, but it was the slowest, probably not worth it.

So, same as OP on 5080, for 5060 Ti, Q4_K_M seems in the sweet spot, balanced between speed and quality.
2
KeldenL
•
1d ago

this is super helpful! totally make it another post. all these quant posts (especially for 16gb, cuz selfishly i also have 16gb vram on my 4060ti) have been super enlightening and saves a lot of people a lot of testing!

i wonder why your t/s was closer to 40 vs OP's 70, cuz that's what i'm seeing too on my end
1
bobaburger
•
1d ago
emoji:Discord:

thank you so much!

the speed difference was due to two things:

    the context window, mine was 128k, OP was 64k (-c 65536)

    OP probably has stronger CPU than mine, and was using 20 threads (-t 20), mine was only 8 threads :D

1
u/mr_Owner avatar
mr_Owner
•
1d ago

Very nice but i believe when you put the pp speed besides them you could make better judgement.
1
u/OsmanthusBloom avatar
OsmanthusBloom
•
1d ago

This is great work! But I wonder about the effect of dropping the batch size adjustments. Normally you increase the ubatch size to improve prompt processing speed. It can increase drastically (eg 3x) when you raise ubatch from, say, 512 to 2048. But generation speed will suffer due to VRAM pressure. You didn't seem to benchmark pp speed separately. Maybe an ubatch size of, say, 1024 would have raised pp without hitting tg too much?
2
u/OsmanthusBloom avatar
OsmanthusBloom
•
1d ago
• Edited 1d ago

Here is my llama-bench result, which shows that increasing ubatch from the default 512 to 1024 or 2048 increases prompt processing speeds a lot, from 280 t/s to 440 and 650 t/s. I have a RTX 3060 Laptop GPU with only 6GB VRAM so most of the model is offloaded to GPU. Using the UD_Q3_K_M quant released today.

llama-bench -m Qwen3.5-35B-A3B-UD-Q3_K_M.gguf -ctk q8_0 -ctv q8_0 --n-cpu-moe 37 -p 4096 -n 512 -fa 1 -b 2048 -ub 512,1024,2048
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6, VMM: yes
| model                          |       size |     params | backend    | ngl | n_ubatch | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | -----: | -: | --------------: | -------------------: |
| qwen35moe ?B Q8_0              |  15.53 GiB |    34.66 B | CUDA       |  99 |      512 |   q8_0 |   q8_0 |  1 |          pp4096 |        283.43 ± 1.24 |
| qwen35moe ?B Q8_0              |  15.53 GiB |    34.66 B | CUDA       |  99 |      512 |   q8_0 |   q8_0 |  1 |           tg512 |         23.90 ± 0.63 |
| qwen35moe ?B Q8_0              |  15.53 GiB |    34.66 B | CUDA       |  99 |     1024 |   q8_0 |   q8_0 |  1 |          pp4096 |        444.65 ± 1.35 |
| qwen35moe ?B Q8_0              |  15.53 GiB |    34.66 B | CUDA       |  99 |     1024 |   q8_0 |   q8_0 |  1 |           tg512 |         23.62 ± 0.25 |
| qwen35moe ?B Q8_0              |  15.53 GiB |    34.66 B | CUDA       |  99 |     2048 |   q8_0 |   q8_0 |  1 |          pp4096 |        648.85 ± 2.56 |
| qwen35moe ?B Q8_0              |  15.53 GiB |    34.66 B | CUDA       |  99 |     2048 |   q8_0 |   q8_0 |  1 |           tg512 |         23.37 ± 0.35 |

build: ecbcb7ea9 (8179)

llama-bench doesn't support --fit so I had to set --n-cpu-moe manually according to the VRAM requirements of the largest ubatch size. With a smaller ubatch size and --fit, some more experts would fit in VRAM and thus generation speeds would be slightly higher. Still, getting much higher pp speeds is I important especially for agentic stuff where prompts can be quite long.
1
