import soundfile as sf
import numpy as np
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5")

# Non-streaming
wav = model.generate(
    text="""When Paul was twenty-five, she gave birth to Frank, her son by Freud. Motherhood is the perfect stage on which to perform feminine masochism, but as many new mothers have discovered, it can also be the occasion to confront one’s inner St Brigid once and for all. Of the many wonderful images in this book, the one I kept returning to isn’t a painting at all but a photo of mother and child, beaming at each other, locked in intense, joyful relation. (Lucian was nervous about holding Frank, ‘found it difficult to deal with the fact that I was preoccupied and not so readily available after the birth’, and, despite already having fathered many children, still appeared confused by the process of raising them: he ‘was disturbed by the milk that had leaked onto my dress: “What’s that?” he asked. I sensed that it repelled him.’) So intense are the joys that Paul experiences her love for her son as a kind of self-submerging, as she notes in her diary, directly after his birth""",
    prompt_wav_path='/Volumes/T7/piper/whisperx/conor/out/wavs/clip1356.wav',
    prompt_text="""Some argue that the Dehaene-Shanjou model describes the broadcasting of information, but does not fully explain why broadcasting should feel like subjective experience.""",
    cfg_value=2.0,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
    inference_timesteps=10,   # LocDiT inference timesteps, higher for better result, lower for fast speed
    normalize=False,           # enable external TN tool, but will disable native raw text support
    denoise=False,             # enable external Denoise tool, but it may cause some distortion and restrict the sampling rate to 16kHz
    retry_badcase=False,        # enable retrying mode for some bad cases (unstoppable)
    retry_badcase_max_times=3,  # maximum retrying times
    retry_badcase_ratio_threshold=6.0, # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
)

sf.write("output.wav", wav, model.tts_model.sample_rate)
print("saved: output.wav")

exit(0)
# Streaming
chunks = []
for chunk in model.generate_streaming(
    text="""The storm, of course, was Freud himself, father of fourteen acknowledged children with six women. Traditionally, museography has considered critiques of such arrangements tediously puritan. (Although, by the time Freud died, in 2011, a grudging shift was taking place. You can hear it in some of the posthumous profiles: ‘As raffishly bohemian as these arrangements may sound, it was no easy road for the women and children involved.’) Interior W11 aestheticizes the complex family romance Freud liked to cultivate around himself. (He seemed to require, Paul delicately suggests, an ‘undercurrent of jealousy…as a stimulus to his own affairs’.) It’s less a group portrait than a staged drama about power, depicting five atomized people tethered to a central figure not pictured. They are all there for Lucian, only for him. Of course, this is true of all portraits – the sitters always appear at the artist’s bidding – but few painters have put as much emphasis on sitting-as-subjection. (The youngest child looks less like a child reclining than a rag doll, destined to remain wherever you drop her.) A year into their relationship, Paul discovered that Freud had several young lovers at the Slade. Devastated, she cautiously expressed her pain to Gowing, he of the kind letter about unpainted paintings. But by now, Celia was a muse, so a different kind of advice was in order: He says that he has known Lucian since he was sixteen and knows that he just doesn’t commit himself to one single person – not through unconcern but just because this was the stony cold mode of living that his art flourished on. Paul records her subsequent suicide attempt with surreal British restraint, in five sentences, never to be mentioned again: Everyone at the Slade was gossiping about Lucian, and they delighted in telling me about all the people he was having affairs with. I became severely depressed. One night I swallowed a packet of Veganin [a painkiller], washed down by a bottle of whisky. This landed me in hospital. I went home to recover. After this she remains with him. Anyone who has ever waited for a text or an email from a lover – but never known the pain of the landline or the postal service – will marvel at the old ways, when a woman could find herself constrained to the house for days, awaiting a sign. And he remains promiscuous. Sometimes he gaslights her about it (‘You’re crazy’). Sometimes he defends it (‘It doesn’t alter what I feel for you’). Sometimes he claims the bohemian’s licence (‘I don’t know if it’s right and I don’t know if it’s wrong’). Sometimes he forgets what he said before and has to adapt""",
    prompt_wav_path='/Volumes/T7/piper/whisperx/conor/out/wavs/clip1356.wav',
    prompt_text="""Some argue that the Dehaene-Shanjou model describes the broadcasting of information, but does not fully explain why broadcasting should feel like subjective experience.""",
    # prompt_wav_path='/Volumes/T7/VoxCPMANE/npy/fraser.wav',
    # prompt_text="""He was smallish and spare, with brown hair rather too regularly waived and a strong brown beard cut to a point. His double-breasted suit of navy blue and his socks, tie, and handkerchief, all scrupulously matched, were a trifle more point-device than the best taste approves.""",
    # prompt_wav_path='/Volumes/T7/VoxCPMANE/npy/jessica.wav',
    # prompt_text="Welcome to the world of imagination, where every voice brings a new character to life. Today, I'll be transforming into a mischievous goblin, a wise old wizard, and even a daring space explorer.",
    cfg_value=2.0,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
    inference_timesteps=10,   # LocDiT inference timesteps, higher for better result, lower for fast speed
    normalize=False,           # enable external TN tool, but will disable native raw text support
    denoise=False,             # enable external Denoise tool, but it may cause some distortion and restrict the sampling rate to 16kHz
    retry_badcase=False,        # enable retrying mode for some bad cases (unstoppable)
):
    chunks.append(chunk)
wav = np.concatenate(chunks)

sf.write("output_streaming.wav", wav, model.tts_model.sample_rate)
print("saved: output_streaming.wav")