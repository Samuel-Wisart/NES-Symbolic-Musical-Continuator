"""
protocol

Protocolo de avaliação e geração compartilhado entre o Transformer e o Markov.

Avaliação (perplexidade):
    perplexidade = exp(média da entropia cruzada, em log natural, de todos os
    tokens depois do BOS em cada música de teste). São TEST_TARGET_TOKENS
    tokens, a soma de len(ids) - 1 nos .npz de teste.

Geração:
    Para cada música de PROMPT_SONGS (nesta ordem), o prompt são os primeiros
    PROMPT_SECONDS segundos dos tokens de teste, cortados exatamente nesse ponto
    (um TIME_SHIFT que atravessa o corte é dividido), sem EOS e com as notas
    abertas continuando abertas. A continuação tem exatamente GEN_SECONDS
    segundos: EOS não é permitido antes de MIN_GEN_SECONDS (= GEN_SECONDS) e
    nenhum TIME_SHIFT pode passar do limite. Amostragem com TEMPERATURE e
    TOP_K, semente generation_seed(i) para a i-ésima música. No fim, as notas
    abertas são fechadas.

    A máscara de eventos válidos (nes_continuation.sampling.allowed_tokens) não
    depende do modelo e pode ser usada pelo Markov:
      - nunca PAD nem BOS;
      - NOTE_ON só em voz sem nota soando;
      - NOTE_OFF só para (voz, pitch) aberto, e não no passo do próprio NOTE_ON;
      - EOS só depois de MIN_GEN_SECONDS; TIME_SHIFT sem passar de GEN_SECONDS.
    Utilitários do corte e do fechamento: nes_continuation.representation
    (cut_prompt, close_open_notes, write_midi).
"""

### Imports
from pathlib import Path

### Caminhos
ROOT = Path(__file__).resolve().parents[1]
TEST_TOKENS_DIR = ROOT / "data" / "processed" / "nesmdb" / "tokens" / "test"
GENERATED_DIR = ROOT / "results" / "generated"

### Avaliação
TEST_SONGS = 372
TEST_TARGET_TOKENS = 435683

### Geração
PROMPT_SECONDS = 4
GEN_SECONDS = 12
MIN_GEN_SECONDS = GEN_SECONDS
N_PROMPTS = 50
SEED = 42
TEMPERATURE = 1.0
TOP_K = 32

# Quantas continuações também viram WAV (as primeiras da lista).
N_WAV = 5

# Sorteadas uma vez com random.Random(SEED).sample(elegíveis, N_PROMPTS), onde
# elegíveis = nomes (em ordem alfabética) das músicas de teste com pelo menos
# PROMPT_SECONDS + 2 segundos (259 músicas). Ficam fixas aqui; o
# tests/test02_transformer_tests.py confere que a lista segue essa regra.
PROMPT_SONGS = (
    "071_DigitalDevilStory_MegamiTensei_11_12MazurkaMazurkaCorridor",
    "002_1943_TheBattleofMidway_12_13AttackTarget",
    "183_Karnov_05_06FinalBattle",
    "164_Hogan_sAlley_00_01TitleScreen",
    "389_WreckingCrew_04_05GoldenHammer",
    "037_BoobyKids_07_08BoobyMarch",
    "256_OperationWolf_01_02OperationStart",
    "016_ArkanoidII_02_03Introduction",
    "241_MiracleRopit_sAdventurein2100_02_03ThemeofGirl",
    "256_OperationWolf_02_03MissionAccomplished1",
    "290_Seicross_03_04BGM2",
    "183_Karnov_04_05HurryUp",
    "013_AlterEgo_07_08Congratulations",
    "194_KnightMove_08_092PlayersWin",
    "145_Golgo13_TopSecretEpisode_16_17EveChristy",
    "002_1943_TheBattleofMidway_08_09AntishipBattleC",
    "002_1943_TheBattleofMidway_07_08AirBattleC",
    "013_AlterEgo_08_09GameOver",
    "071_DigitalDevilStory_MegamiTensei_09_10ValhallaValhallaCorridor",
    "071_DigitalDevilStory_MegamiTensei_13_14HellfireCorrodingSeaofFlames",
    "176_InsectorX_03_04PlateauAreaRound2",
    "214_MadoolanoTsubasa_03_04MainUnderground",
    "002_1943_TheBattleofMidway_06_07FailedintheAttack",
    "194_KnightMove_00_01MenuOption",
    "071_DigitalDevilStory_MegamiTensei_03_04DaedalusDaedalusTower",
    "242_MississippiSatsujinJiken_05_06NelsonsTheme",
    "237_MightyBombJack_08_09Sky",
    "241_MiracleRopit_sAdventurein2100_08_09ThemeofMiracleRopitunusedversion",
    "361_Transformers_ConvoynoNazo_08_09Unknown",
    "145_Golgo13_TopSecretEpisode_14_15HerodesConcertHall",
    "071_DigitalDevilStory_MegamiTensei_10_11PowerAttackBoss",
    "145_Golgo13_TopSecretEpisode_23_24Unused2",
    "194_KnightMove_07_08Congratulations",
    "085_DragonBusterII_YaminoFuuin_07_08Invincible",
    "285_Route16Turbo_02_03Normal",
    "286_Rygar_14_15Lapis",
    "002_1943_TheBattleofMidway_01_02PowerUpYourP38",
    "265_PokerJ_ngl_ng_01_02PokerJinglingTitleScreen",
    "285_Route16Turbo_01_02Easy",
    "038_BubbleBobble_02_03SecretRoom",
    "241_MiracleRopit_sAdventurein2100_07_08ThemeofBonusZone",
    "361_Transformers_ConvoynoNazo_04_05UCube",
    "116_FelixtheCat_08_09Northpole",
    "286_Rygar_19_20Ending",
    "038_BubbleBobble_00_01IntroductionMainTheme",
    "338_TerraCresta_09_10EditMode",
    "265_PokerJ_ngl_ng_04_05AVPokerMainBGM1",
    "116_FelixtheCat_07_08Tropics",
    "379_WardnernoMori_00_01TitleScreen",
    "338_TerraCresta_10_11Unknown",
)

### Funções
def prompt_paths():
    """Caminhos dos .npz de teste usados como prompt, na ordem do protocolo."""
    return [TEST_TOKENS_DIR / f"{name}.npz" for name in PROMPT_SONGS]

def generation_seed(index):
    """Semente da amostragem da index-ésima música de PROMPT_SONGS."""
    return SEED + index
