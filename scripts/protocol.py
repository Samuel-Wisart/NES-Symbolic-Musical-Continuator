"""Define o protocolo compartilhado de avaliação e geração dos modelos."""

### Imports
from pathlib import Path

### Caminhos
ROOT = Path(__file__).resolve().parents[1]
TEST_TOKENS_DIR = ROOT / "data" / "processed" / "nesmdb" / "tokens" / "test"
# Resultados entregues (versionados): um subdiretório por modelo e as amostras geradas.
RESULTS_DIR = ROOT / "resultados"
GENERATED_DIR = RESULTS_DIR / "amostras"
REFERENCE_NAME = "referencia"
MODELS = ("transformer", "lstm")

def checkpoint_path(model):
    """Retorna o checkpoint local ou o modelo entregue em resultados/."""
    trained = ROOT / "runs" / model / "best.pt"
    return trained if trained.exists() else RESULTS_DIR / model / "modelo.pt"

### Avaliação
TEST_SONGS = 372
TEST_TARGET_TOKENS = 435683

### Geração
PROMPT_SECONDS = 4
GEN_SECONDS = 12
MIN_GEN_SECONDS = GEN_SECONDS
N_PROMPTS = 100
SEED = 42
TEMPERATURE = 1.0
TOP_K = 32

# Quantas continuações também viram WAV (as primeiras da lista).
N_WAV = 5

# Seleção reproduzível entre músicas com pelo menos PROMPT_SECONDS + 2 s.
# A lista fixa mantém os mesmos prompts entre execuções.
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
    "145_Golgo13_TopSecretEpisode_03_04Cutscene1ActionStage4",
    "015_Arkanoid_03_04Ending",
    "124_FleetCommander_00_01Title",
    "116_FelixtheCat_09_10Space",
    "338_TerraCresta_05_06Dycoon",
    "085_DragonBusterII_YaminoFuuin_01_02Demo",
    "002_1943_TheBattleofMidway_11_12Ayako",
    "242_MississippiSatsujinJiken_10_11MainThemeExtended",
    "158_HikarinoSenshiPhoton_WakuseiZoldiasnoTatakai_09_10Area23",
    "183_Karnov_00_01Opening",
    "037_BoobyKids_02_03BoobySamba",
    "145_Golgo13_TopSecretEpisode_02_03MainTheme",
    "013_AlterEgo_05_06Level4",
    "183_Karnov_06_07AllCleared",
    "085_DragonBusterII_YaminoFuuin_12_13NorthernEuropeRound6",
    "231_MetalFighter__01_02ItemBGMEndingBGM",
    "214_MadoolanoTsubasa_08_09Ending",
    "124_FleetCommander_01_02Ending",
    "194_KnightMove_04_05Umasakaba",
    "071_DigitalDevilStory_MegamiTensei_02_03MicomStartingMicomCity",
    "242_MississippiSatsujinJiken_00_01MainTheme",
    "013_AlterEgo_02_03Level2",
    "272_Quarth_02_03TheMenaceoftheBlock",
    "237_MightyBombJack_11_12Ending",
    "071_DigitalDevilStory_MegamiTensei_12_13PrayJakyou",
    "085_DragonBusterII_YaminoFuuin_10_11WilderAppearance",
    "265_PokerJ_ngl_ng_05_06AVPokerMainBGM2",
    "338_TerraCresta_08_09MarchingRaster",
    "016_ArkanoidII_01_02Story",
    "285_Route16Turbo_03_04Difficult",
    "286_Rygar_05_06IndoraGodJPVersion",
    "158_HikarinoSenshiPhoton_WakuseiZoldiasnoTatakai_02_03Area12",
    "237_MightyBombJack_01_02MainBGM",
    "128_FormationZ_00_01TitleScreen",
    "038_BubbleBobble_03_04FalseEnding",
    "141_Ghostbusters_01_02MainBGM",
    "116_FelixtheCat_12_13Ending",
    "071_DigitalDevilStory_MegamiTensei_07_08WindowLightShop",
    "085_DragonBusterII_YaminoFuuin_03_04ForestRound12",
    "237_MightyBombJack_05_06Labyrinth",
    "013_AlterEgo_03_04Level2Alternate",
    "214_MadoolanoTsubasa_04_05MainTowerCastle",
    "241_MiracleRopit_sAdventurein2100_05_06ThemeofUniverse",
    "038_BubbleBobble_05_06RealEnding",
    "176_InsectorX_12_13NameEntry",
    "071_DigitalDevilStory_MegamiTensei_16_17AnfiniAnfiniPalace",
    "241_MiracleRopit_sAdventurein2100_03_04ThemeoftheEnemyBaseandCave",
    "158_HikarinoSenshiPhoton_WakuseiZoldiasnoTatakai_13_14Area32",
    "308_StarForce_02_03Purser",
    "085_DragonBusterII_YaminoFuuin_05_06DragonMountainClearDemo",
)

### Funções
def prompt_paths():
    """Caminhos dos .npz de teste usados como prompt, na ordem do protocolo."""
    return [TEST_TOKENS_DIR / f"{name}.npz" for name in PROMPT_SONGS]

def generation_seed(index):
    """Semente da amostragem da index-ésima música de PROMPT_SONGS."""
    return SEED + index
