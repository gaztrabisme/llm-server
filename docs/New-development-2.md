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
