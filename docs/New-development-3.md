r/LocalLLaMA
•
1mo ago
lemon07r
llama.cpp
16 GB VRAM users, what model do we like best now?
Discussion

I'm finding Qwen 3.5 27b at IQ3 quants to be quite nice, I can usually fit around 32k (this is usually enough context for me since I dont use my local models for anything like coding) without issues and get around 40+ t/s on my RTX 4080 using ik_llama.cpp compiled for CUDA. I'm wondering if we could maybe get away with iq4 quants for the gemma 26b moe using turboquant for kv cache..

Being on 16gb kind of feels like edging, cause the quality drop off between iq4 and q4 feel pretty noticable to me.. but you also give-up a ton of speed as soon as you need to start offloading layers.
218
u/CodeRabbitAI avatar CodeRabbitAI
•
Promoted
When did your code reviewer last catch an edge case? CodeRabbit catches them before they ship.
Sign Up
coderabbit.link
Thumbnail image: When did your code reviewer last catch an edge case? CodeRabbit catches them before they ship.
Sort by:
Comments Section
sine120
•
1mo ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Like you said, 27B at IQ3_XXS does well. I have 64GB of system RAM, so I tend to run MoE's in harnesses with a small amount of system prompt if possible. Qwen3-Coder is good, 3.5-35B-A3B is good, and Gemma4-26B is good. If I don't need as much intelligence/ coding ability, 3.5-9B is also pretty good, and I want to play with Qwopus to see how it handles.

I wish there were something up-to-date in the 12-20B range, as that would probably give 16GB folks enough context to be more useful and use higher quants.
71
xeeff
•
1mo ago

please let me know how Qwopus (9B/35B A3B/27B) works out for you, and what your use cases are. i'll be waiting :)
7
sine120
•
1mo ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Is there a 35B Qwopus? I only see 4/9/27B.
2
u/grumd avatar
grumd
•
1mo ago
• Edited 1mo ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

You should try 122B at IQ3_S, at a low quant it outperforms 27B. 27B gets ahead of 122B at higher quants
8
u/Big-Wear-8148 avatar
Big-Wear-8148
•
1mo ago

how would it fit 16gb vram ?
2
u/grumd avatar
grumd
•
1mo ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

It doesn't need to. It's a MoE model, experts can be offloaded to CPU/RAM
3
u/IrisColt avatar
IrisColt
•
1mo ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Your comment is really helpful. I have 24GB of VRAM and 64GB of RAM as well, but I need to fit 128K of context with Gemma 4 31B Q4_K_M. (For Qwen 3.5 27B, 256K is the practical limit without a noticeable speed hit.) With Gemma 4, though, it's essentially impossible to reach 64K context without running into RAM speed penalties.

Is Gemma 4 at IQ3_XSS performing well too?
3
u/IrisColt avatar
IrisColt
•
1mo ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

I checked and IQ3_XSS with Gemma4 is sadly unusable...
1
u/hideo_kuze_ avatar
hideo_kuze_
•
1mo ago

    With Gemma 4, though, it's essentially impossible to reach 64K context without running into RAM speed penalties.

Even with the recently dropped TurboQuant improvement?
1
u/ansibleloop avatar
ansibleloop
•
1mo ago

My issue is I want the entire model in my GPU for speed, but with my monitors I only have like 15GB of RAM with 12GBish for the model and 3GB for context

I need to offload some of that and try Gemma 4
2
sine120
•
1mo ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

For low context/ quick chats, you can fit pretty good intelligence in 16GB, but for longer context work you'll pretty much need to give up on that and accept it's going to be a background task.
3
u/FullOf_Bad_Ideas avatar
FullOf_Bad_Ideas
•
1mo ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

I'd try 3.10bpw EXL3 quant

https://huggingface.co/UnstableLlama/Qwen3.5-27B-exl3-3.10bpw
1
reery7
•
1mo ago
• Edited 1mo ago

Mhm, the 27B IQ3_XSS is just okayish. Q3 is somewhat always a last resort. For visual input I usually test with a picture to identify a species and the IQ3_XXS fails miserably.
The qwen3-vl-8b-instruct (8 bit quant, MLX for me) is way better in that regard, almost twice as fast as well. Qwen3.5 27B distilled 4 bit quant is also a significant step up, but not usable on 16 GB VRAM.
1
Spicy_mch4ggis
•
1mo ago

Qwopus is pretty decent, I quite like it
1
Morphon
•
1mo ago

Qwen 3.5 over here!

35B-A3B at Q6K and 128k context (expert weights pushed to CPU). 35t/s. Very usable speeds, low precision loss because of the big quant.

122B-A110B at IQ3_S and 128k context (again, expert weights on CPU). 15t/s. Still usable speeds, but not as "Just ask the AI and get an answer right away" level of speed. Less precision, but MUCH better domain knowledge.

These two have replaced almost everything else I've used.
45
u/n8mo avatar
n8mo
•
1mo ago

Agreed.

35B-A3B is by farrrr my favourite model atm. Fast enough on my 5070ti and smart enough for most things I use it for.

That said, I haven’t had a chance to toy with the new Gemma models.
14
u/OneStoneTwoMangoes avatar
OneStoneTwoMangoes
•
1mo ago

What quant of Qwen 35 runs well on 5070Ti Laptop?
1
u/Kahvana avatar
Kahvana
•
1mo ago
emoji:hf:
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Give it a try when you can, they both compliment each other well
1
u/toalv avatar
toalv
•
1mo ago

How do you push expert weights to CPU? I'm using Ollama, does it do this automatically or do I need to use llama.cpp or similar?
3
u/mlhher avatar
mlhher
•
1mo ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

You should stop using Ollama and use llama.cpp. You do not need any config for llama.cpp just use "-fit on". You can use it for all models, llama.cpp just smartly fits whatever gives the best speed.

Ollama should be avoided for many many reasons.
34
Jayfree138
•
1mo ago

Does llama.cpp swap models in and out of VRAM as needed or do you have to do it manually? With an Ollama backend if i call a different model than the one that is loaded and i dont have enough VRAM to fit both it'll drop the previous model out of VRAM automatically to make space for the one i'm using rather than overflow to system RAM.

This enables me to string together multiple models in a sequence with minimal VRAM usage, which is critical on a consumer GPU with limited memory.

If llama.cpp can do that with minimal setup i'll seriously think about switching.
3
fligglymcgee
•
1mo ago

Yes, llama.cpp now has a release called llama-server that handles this pretty well. Llama-swap is a bit more flexible, but either are good choices and both will hot swap models on demand for you.
2
Jayfree138
•
1mo ago

Thanks, I'll check that out!
1
u/lolwutdo avatar
lolwutdo
•
1mo ago

I thought -fit was on by default?
2
Morphon
•
1mo ago

I'm using LMStudio. Not sure what the actual flags would be if running this on the cmdline.

This allows me to use my 64GB of system RAM to circumvent the speed tax on these bigger models. KV Cache and some layers sit on the GPU. Inference experts sit in RAM and are partially run on the CPU.

It's been a huge game changer for me.
4
u/toalv avatar
toalv
•
1mo ago

I'm on 64GB as well, appreciate the tips.
1
u/DragonfruitIll660 avatar
DragonfruitIll660
•
1mo ago

I think Ollama can use n-cpu-moe to offload experts to regular ram. If I remember right there is a slider for it (I haven't really used Ollama, generally just use llama.cpp but I remember hearing about it)
1
Di_Vante
•
1mo ago

How did you get 122b properly configured? Did you set like specific params, or are using stock? I'm only getting trash from it :(
1
AvidCyclist250
•
1mo ago
llama.cpp

I use this, with models downloaded via lm studio.

cd llama.cpp ./build/bin/llama-server
-m "/home/-----YOURNAME----/.lmstudio/models/unsloth/Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-UD-IQ3_XXS.gguf"
--jinja
-b 2048 -ub 2048
--temp 0.6
-ctk q8_0 -ctv q8_0
-fitc 68304 --fit on -fitt 256
--cache-ram 0 --parallel 1
-t 6 --reasoning-budget 1024 \
7
Di_Vante
•
1mo ago

Awesome, ill try this out. Tyvm!
1
AvidCyclist250
•
1mo ago
llama.cpp

Can lower q8 to to q4 for more context. Might want to test your use cases before doing that. Haven't noticed any big drawbacks, but others have said it made a difference for them
1
u/iamapizza avatar
iamapizza
•
1mo ago

Could you share your llama server arguments for both setups, might help for comparison. I never thought to run a 122b on a 16gb card. 
1
u/Popular_Tomorrow_204 avatar
Popular_Tomorrow_204
•
1mo ago

Im a complete beginner, so i might not understand correctly.

    35B-A3B at Q6K and 128k context (expert weights pushed to CPU). 35t/s. Very usable speeds, low precision loss because of the big quant.

Are you using it for coding or other Tasks? If yes would you recommend it for coding?
1
Morphon
•
1mo ago

I don't use it for "vibe coding". But I will ask it questions about syntax and standard library functions, and occasionally for some code review tasks (how can I make this function more efficient/idiomatic, etc...). If it has good training data for the language (like JavaScript, Python, etc...), it does quite well for these tasks. Rarer languages (Smalltalk) - not so great. It will hallucinate methods like you wouldn't believe! :-)

EDIT: I should add - I don't use it for vibe coding because I don't vibe code. I have no idea how useful it is for that task, or how it would compare to a big datacenter-sized model.
2
u/Monad_Maya avatar
Monad_Maya
•
1mo ago
llama.cpp
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Why IQ3_S on 122B, system specs? 
1
LoSboccacc
•
1mo ago

What the prompt processing speed of that?
1
u/-Ellary- avatar
-Ellary-
•
1mo ago
emoji:Discord:
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

5060ti 16gb.

gemma-4-26B-A4B-it-IQ4_XS - 90k of context (Q8) - all layers - 90tps.
gemma-4-31B-it-IQ4_XS - 16k of context (Q8) - 52 layers - 10tps.
gemma-4-31B-it-IQ3_XXS - 45k of context (Q8) - all layers - 25tps.
Qwen3.5-27B-IQ4_XS - 20k of context - all layers - 25tps.
Qwen3.5-27B-heretic-v3.i1-IQ3_XXS - 77k of context - all layers - 25tps.
Skyfall-31B-v4.2-IQ3_XXS - 32k of context (Q8) - all layers - 25tps.

IQ3_XXS is surprisingly good, It is around Q2K size but performance is really better.
I'd say there is just no point of running 9b model at Q8, just run IQ3_XXS 27b, size is the same.
16
InternationalNebula7
•
1mo ago

This is a very helpful post.

5080 16gb; no vision

gemma-4-31B-it-Q3_K_S - 18994 context (Q8) - all layers - tg 45tps, pp 1577tps
gemma-4-31B-it-Q3_K_M.gguf - 18k context (Q8) - 55 layers - tg 17.5 tps, pp 1100tps
gemma-4-26B-A4B-it-UD-IQ4_NL.gguf - 18k context (Q8) - all layers - tg 136tps, pp 5567tps
4
u/Richardbobbyryan1125 avatar
Richardbobbyryan1125
•
1mo ago

Just the exact comment I was looking for , 5060ti 16gb , you're the 1st person I found that actually uses the same as mine, perfectly helpful man
4
u/scrapedo_ avatar u/scrapedo_
•
Promoted
No more blocks, no more scraper downtime. Meet the better, faster, stronger web scraping API. Scrape.do beats competition every time.
Learn More
scrape.do
Thumbnail image: No more blocks, no more scraper downtime. Meet the better, faster, stronger web scraping API. Scrape.do beats competition every time.
u/the__storm avatar
the__storm
•
1mo ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

I've been using Gemma 4 26B at IQ4_XS; gets about 65K context at fp16. I agree that the IQ4 is more compressed than I'd like, but I find that Gemma is still quite good at non-coding tasks.

I have 64GB system memory but it's dual channel DDR4 so I'm loathe to offload anything with lots of active parameters to it. If there was an updated Coder-Next (80B-A3B) that would be a nice option.
14
Herocem
•
1mo ago

Gemma 4 26B-A4B for me at Q4, 128k. I get 60/ts when context is empty and goes all the way down to 40 t/s when it gets full. I am running it on 5070 ti, 32 gb ddr4 3600, Ryzen 7 5800X3D.

I use it for my personal assistant project on n8n.
13
lemon07r
OP •
1mo ago
llama.cpp

Hmm I want to try this, but at the same time that only marginally faster than dense 27b at iq3.. and I get the feeling a dense 27b model would still be smarter and more capable.
1
u/k0valik avatar
k0valik
•
1mo ago

If it's not too much hassle can you please share your llama.cpp config? I have the exact same config (except CPU is intel counterpart), but I struggle to fit into 16gb, although I generally have 15gb vram available due to heavy frontend apps and system apps (also shamefully admitting I'm running on windows)
1
Herocem
•
1mo ago
• Edited 1mo ago

Yup:
.\llama-server.exe -m .\gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf -c 131072 --fit on -fa on -t 8 --no-mmap --jinja -ctk q8_0 -ctv q8_0 --temp 0.8 --top-p 0.95 --top-k 64

And this is my qwen 3.5 35b a3b config:
.\llama-server.exe -m .\Qwen3.5-35B-A3B-Q4_K_M.gguf -c 65536 --fit on -fa on -t 8 --no-mmap --jinja -ctk q8_0 -ctv q8_0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 1.5 --repeat-penalty 1.0
4
u/Ps3Dave avatar
Ps3Dave
•
1mo ago

    And this is my qwen 3.5 35b a3b config

Thank you so much. I don't know why but I was having trouble running qwen 3.5 in my setup (12GB VRAM + 32GB RAM) but with your command line it worked perfectly. Getting 42t/sec even with 256k context, and fitting in about 63% of RAM.
2
ea_man
•
1mo ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

If you don't waste VRAM you should be able to fit Qwen_Qwen3.5-27B-IQ4_XS.gguf 15.2 GB with some 80k spare context at Q_4.

- https://huggingface.co/bartowski/Qwen_Qwen3.5-27B-GGU

Either use integrated graphics for DE or kill X11, otherwise if you tune it properly you should be able to run LXqt with some 40k context.

BTW: Qwen_Qwen3.5-27B-IQ3_XXS.gguf 11.3 GB runs the same way on a 12GB GPU.
11
u/Top-Rub-4670 avatar
Top-Rub-4670
•
1mo ago

That's insane, how do you fit 15.2GB in 16GB VRAM? Where is the KV cache going? The context? Hell, your OS' desktop renderer?

I guess you just have a secondary card that is dedicated to running the IQ_XS 15.2GB? With all types of caching disabled?
4
[deleted]
•
1mo ago

u/Witty_Mycologist_995 avatar
Witty_Mycologist_995
•
1mo ago

Gemma 26b all the way
32
u/throwaway957263 avatar
throwaway957263
•
1mo ago
• Edited 1mo ago

What quant did you use? I tried https://ollama.com/VladimirGav/gemma4-26b-16GB-VRAM

But it leaves you with 1GB VRAM for KV cache, leaving you with 8192 context

E: now seeing 8k might be ollama config issue
3
u/random_boy8654 avatar
random_boy8654
•
1mo ago

Gemma 26b vs qwen 3.5 35B ?
2
u/Witty_Mycologist_995 avatar
Witty_Mycologist_995
•
1mo ago

Gemma.
4
LostDrengr
•
1mo ago

Currently using Gemma4-26B-IQ3 plenty of room for context and its hitting 124t/s on 5080.
10
u/Long_comment_san avatar
Long_comment_san
•
1mo ago
emoji:hf:
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Damn can't wait to get a reasonably priced GPU with 32 gb VRAM. R9700 is quite close as is B70, but nah, I do play games as well. No idea why AMD doesn't just click it and push something in the 800$ with 24gb with slower VRAM. Running AI with 12 and 16gb is fucking miserable.
9
ea_man
•
1mo ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Aye, I want a 9070xt Super with double VRAM.
2
Maleficent_Celery_55
•
1mo ago

i wish amd did something like 7900xtx this generation. i hope they do it next time.
2
lly0571
•
1mo ago

Gemma4-26B-A4B-IQ4_XS for speed and Qwen3.5-27B-Q3_K_XL for quality. Both of them can handle ~32k context with 16GB.
7
u/Top-Rub-4670 avatar
Top-Rub-4670
•
1mo ago

I haven't noticed any difference between Q3_K_XL and Q3_K_M and the benchmarks seem to agree. Has your experience been different? I like the (small amount of) extra context I can fit.
1
[deleted]
•
1mo ago

u/Vn88mkt avatar u/Vn88mkt
•
Promoted
Shop Bán thẻ game usdt, chiếu khấu cao, siêu ẩn danh, Telegram: @DLYW88
muathegame.biz
Clickable image which will reveal the video player: Shop Bán thẻ game usdt, chiếu khấu cao, siêu ẩn danh, Telegram: @DLYW88
0:00 / 0:00
Ell2509
•
1mo ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Just FYI there is no edge. Everyone wants the next size up.
5
Sadman782
•
1mo ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter


send-moobs-pls
•
1mo ago

I'm in the 8GB poor house but I just can not find anything that compares to qwen 3.5 models right now. I'll say maybe their weakness is like creativity or role play or something because the qwen vibe is pretty "codex" feeling, Gemma might be better if you specifically want creativity or personality like that. But for general thinking, tasks, tools etc I'm basically still in shock at how the qwen 9B makes everything else I can run look like a joke
3
lemon07r
OP •
1mo ago
llama.cpp

I really need to look into the Gemma models, but I'm not entirely convinced they will be better than the qwen 3.5 models. EQ bench actually shows qwen 3.5 27b model to be the better writer than any of the gemma models.
2
H3g3m0n
•
1mo ago

The IQ3_XXS of Gemma-31b should allow for around 60k context (With Q8 kv cache). Someone posted benchmarks of twitter that it's basically as good as the Q4. Could even get more context with something like turboquant/rotorquant if your willing to figure out which random fork is decent.

Unfurtunatly as of now CUDA 13.2.0 has a bug that causes it to output gibbirish in llama.cpp I tried downgrading to 13.1 which solved the gibberish issue but ran into another bug that caused it to crash if loading the vision mmproj. Might try 13.0 or a 12.x and see if they solve both bugs.

Currnelty I'm just sticking with the MOE of Qwen which gets the full context and decent speeds with n-cpu-moe offloading. It seems better than the Gemma4 MOE.
3
u/AlterTableUsernames avatar
AlterTableUsernames
•
1mo ago

Greath answers here, anyone a recommendation for 8GB VRAM + 32GB DDR4?
3
u/moahmo88 avatar
moahmo88
•
1mo ago

5070ti,32gb ram:
bartowski/Qwen_Qwen3.5-27B-GGUF IQ3_XS ,ctx 131k ,45t/s
HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive q5k_m,ctx 131k,62t/s
3
u/Shamp0oo avatar
Shamp0oo
•
1mo ago

You can run the IQ4_XS quant of Qwen 3.5 27B with 16 GB of VRAM and up to 40k context (q8). See my comment and follow up comment for instructions. I recently switched to the unsloth IQ4_XS quant which is slightly bigger and therefore only allows for around 32k context but it felt more robust with tool calls in Open WebUI to me.
2
u/popcornkiller1088 avatar
popcornkiller1088
•
1mo ago

gemma-4-26B-A4B-it-UD-IQ3_S.gguf is awesome for 16gb Vram ( RTX 4080 Super)

While I can hit 90 t/s at 32k context on the main card, bridging the second PC let me bump the context up to 130k. Speed dropped to 20 t/s, but having that massive window is a total game changer.

Experimenting with llama.cpp RPC servers to bypass VRAM limits. Using an RTX 4080 Super + an RTX 3060 Ti (8GB) via Ethernet.
2
u/ThankGodImBipolar avatar
ThankGodImBipolar
•
1mo ago
• Edited 1mo ago

No AMD representation here yet; I've been able to run Gemma 4 27B Q8 at 15-20 tok/s on my 7800XT (E - seems to run closer to 20tok/s when adding ot exps=CPU to my launch command). I've also tried a Q4_K_M quant (Heretic, if that makes any difference), and that runs at ≈25tok/s. I haven't rebuilt llama.cpp since Gemma 4 came out, so it's possible it may run faster on the current branch. I'm planning on doing some more messing around tonight and may update if I can find some improvements.

In addition to that, I've also been using Qwen 3.5 Coder Next (64GB of system RAM) at IQ4_XS, and that runs at ≈28tok/s. Not sure whether this or Gemma 4 27B is better for coding; will have to experiment some more.

I'd appreciate if anyone has any insight into whether these speeds seem appropriate for my hardware, if I'm using stupid quants, etc.. I'm going to keep following along with this thread.
2
u/zkstx avatar
zkstx
•
1mo ago

I'm getting 60-70 tps TG / 1300 tps PP, up to 55k context window (100k+ @ Q8 KV) with Qwen3.5 35B IQ3_XXS on my RX6800 XT, llama.cpp compiled for rocm. Presumably, IQ3_XXS leads to a certain amount of brain damage but it handles pretty much anything I throw at it pretty well. I can definitely recommend trying a smaller quant that fits fully into VRAM, it's a lot of fun.
1
Equivalent_Bit_461
•
1mo ago
• Edited 1mo ago

Is it even worth using models this low in quant? I just took the moe pill and run everything quant 6, most important bits in gram, rest on ram, also thanks to turbo quant easily can stay over 100k context. Sure quant 6 might not be lighting fast but at least is not severely reduced.

Edit: since I bought ram before the rampocalypse I can easily run even quant 6 120b moe models, with offloading. As much as I would want to run dense models on my 16gb vram gpu, i get faster speed with moes 4-5 times bigger
2
u/Top-Rub-4670 avatar
Top-Rub-4670
•
1mo ago

In my tests IQ3/Q3 has been fine for both Qwen 3.5 27b and Gemma 4 31b. Asking specific questions about some deep knowledge is definitely worse than Q4+, but the reasoning seems to mostly be there? At least it hasn't failed any of my go-to test tasks.

I found that Q3 was "fine" for role playing in Gemma 4 26b, but it doesn't follow directions as well as Q4+ and it tends to get confused in long contexts. It also frequently forget its personality and starts talking neutral. As for Q2 it's the same, but worse, plus it starts making lots of typos. I haven't noticed any significant difference between Q4/Q5/Q6/Q8 for this purpose. So there seems to be a threshold at Q4 for 26b, and possibly for other similarly sized MoE models?

But Q3 for Qwen 3.5 9b and Gemma 4 E4B is like a lobotomy, they fail all the "complex" tasks I've tried.

Note that I have tried all the small quants out there for the models I've talked about. The static ones, the imatrix ones, the unsloth ones. It doesn't make any real difference, the Q3/Q4 cliff is real!
3
InternationalNebula7
•
1mo ago

What speeds with what config and hardware?
1
TastyStatistician
•
1mo ago

Gemma 4 26b is currently the best for 16gb VRAM.

Qwen 3.5 is also great but thinks way too much. 4b or 9b with thinking off are great if you need large context room.
2
MerePotato
•
1mo ago

Unsloths Q6_K quant of Gemma 4 26BA4B with MoE offloading (--n-cpu-moe) is your best bet imo, just make sure you're on the latest build of llama.cpp.
2
Dabalam
•
1mo ago

I understand people getting 60 t/s won't be fretting about their speed, but people using Q3 dense models at 20 t/s could be getting 2 to 3 times the speed with similar quant MoE or the same speed at Q4. I'm surprised the speed difference isn't so important to most.
2
Hyrnos
•
1mo ago

Has anyone tried the Gemma REAP versions ? Like with 20% expert weights pruned
2
taking_bullet
•
1mo ago

You got my attention buddy. I'm gonna try these REAP versions this weekend. 
1
Hyrnos
•
1mo ago

Let me know how it goes !
1
u/Techngro avatar
Techngro
•
1mo ago

Did you try them? How was it?
1
Guilty_Rooster_6708
•
1mo ago

Gemma 26B and Qwen3.5 35B. MoE all the way
2
Fyksss
•
1mo ago

gemma 4 31B Q3_K_S and IQ3_XXS
1
RandomTrollface
•
1mo ago

Gemma 4 31b and qwen 3.5 27b both iq3_xxs. They seem smarter to me than the smaller models at higher quant.
1
u/Danmoreng avatar
Danmoreng
•
1mo ago

Why ik_llama over llama.cpp?
1
lemon07r
OP •
1mo ago
llama.cpp

Supposedly has optimizations that make it faster, which I think upstream ends up getting some of too, but a lot more slowly. Also supports more esoteric gguf quants, if that's your thing. I usually avoid those since they are usually slower to run. Third reason, they support more kv cache quantization schemes, but I believe llamacpp supports almost as many now
1
u/Danmoreng avatar
Danmoreng
•
1mo ago

Well, I tested that a few months ago and found no performance benefits, that’s why I stick to llama.cpp. The only benefit apparently might be different quants (the IQ ones) which llama.cpp won‘t get because of personal differences: https://github.com/ggml-org/llama.cpp/pull/19726#issuecomment-3946355613

If you want to try out llama.cpp, I got some scripts to build from source and settings I found optimal for the Qwen 3.5 family here: https://github.com/Danmoreng/local-qwen3-coder-env

The 27B model in Q4 is too large for 16GB though. I prefer MoE variants, since they have decent performance if split between GPU and CPU. For example I get around ~70 t/s with the 35B model on my RTX 5080 mobile.
1
lemon07r
OP •
1mo ago
llama.cpp

Yeah there seems to be sort of an ebb and flow of llama.cpp catching up, ik having stuff added, etc. I think the gap has gotten pretty small now, but since ik works too I havent had a reason not to use it. It does compile a little slower though
1
u/AlwaysLateToThaParty avatar
AlwaysLateToThaParty
•
1mo ago

Qwen3.5 9B Heretic Q6_K or Q8_0, depending how much else i have in VRAM. My work computer is locked down. Can't even plug a phone into it to charge it. But at least it has an RTX 5000 in it. So that's what I use if I need to use inference at work. Not as good as my home system, but it works a treat.
1
embeeweezer
•
1mo ago

I'm in the qwen3.5 35b MoE ballpark as well. Would like to get the 27B up to speed though. Anyone got a Speculative Decoding config running?
1
Enough_Big4191
•
1mo ago

16gb still feels like the most annoying tier because one small step up in quant and suddenly the whole setup gets worse. i keep coming back to models that fully fit and stay fast, because once i start offloading layers the quality gain usually stops feeling worth it.
1
lemon07r
OP •
1mo ago
llama.cpp

I had an 8gb card for years then a 12gb one for a few years too, before recently getting back 16gb. Trust me, those were more annoying tiers to be in.
1
Monkey_1505
•
1mo ago
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Can I ask why you are using the ik fork?
1
celebrateurmom
•
1mo ago

I'm using Allen Institute (AI2) OLMo 32b Instruct on my NvMe and OLMo 7b Think on my 4060ti 16Gb. Very cognitive and 5-7 tokens/ sec. Very Satisfied 💯
1
u/4xChe avatar
4xChe
•
1mo ago

Qwen3.5:35b A3B here, running on two 6600XT's... not perfect but enough for my local needs. Tried different engines: ollama, llama cpp, lm studio.... Best speed I got is on Kobold at roughly 25 t/s
1
u/OlegDoDo avatar
OlegDoDo
•
1mo ago

Running qwen2.5:7b on a Snapdragon ARM64 laptop, 16GB RAM, CPU only. Getting around 20–40 sec per response — not instant, but totally usable for document work. For 8GB I'd go gemma3:4b, runs noticeably faster. Both through Ollama + AnythingLLM, no Docker needed.
1
u/moflinCASIO avatar
moflinCASIO
•
12d ago

I actually just spent the last few hours testing this exact problem on a much weaker setup than a 4080, and honestly I came away way more impressed with 16GB VRAM than I expected.

My setup:

    RTX 4060 Ti 16GB

    i5-11400F (6C/12T)

    32GB DDR4-3200

    llama.cpp CUDA build

    Flash Attention enabled

I used to just run default Ollama setups without really understanding quantization differences, but after compiling llama.cpp properly and testing IQ quants directly, the performance difference was honestly massive.

What surprised me most:
IQ quants absolutely dominated K-quants on my setup.

I tested:

    Gemma4 E4B IQ4_XS

    Gemma4 26B A4B UD-IQ4_XS

    Qwen3.6 35B A3B UD-IQ2_M

    Qwen3.6 35B A3B UD-IQ3_XXS

    Qwen3.6 27B Q3_K_M

Results were kinda shocking to me:

    Qwen3.6 35B A3B UD-IQ2_M -> ~81 tok/s -> ~13.1GB VRAM

    Qwen3.6 35B A3B UD-IQ3_XXS -> ~74 tok/s -> ~14.6GB VRAM

    Gemma4 26B A4B UD-IQ4_XS -> ~61 tok/s -> ~15.7GB VRAM

But then Qwen3.6 27B Q3_K_M only got me ~18 tok/s despite the GPU sitting at 99% utilization the whole time and pulling ~160W.

That was the moment I realized:
K-quants are probably just too compute-heavy for this class of GPU.

So at least on a 4060 Ti, IQ quants felt WAY better than I expected.

And honestly, I kinda agree with your point about 16GB “feeling like edging” lol. The difference between “fully GPU resident” and “slightly overflowing VRAM” is absolutely brutal.

But I also came away thinking:
16GB is actually still a really good place to be if you optimize carefully and stay realistic about quant choice.

Before this I honestly thought “maybe 24GB is basically mandatory now,” but after testing llama.cpp CUDA + IQ quants properly, I’m way less convinced.
1
