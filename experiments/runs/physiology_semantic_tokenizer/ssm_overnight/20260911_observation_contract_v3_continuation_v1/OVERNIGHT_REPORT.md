# Observation-contract diagnostic v3

Feature-layer missingness; original 72 training identities only. O2 is Gaussian MAP. Teacher qualification: none. MAP intervals: NOT_ESTIMATED.

Per-cell result.json and trajectory files own the evidence. Tables below retain the complete planned denominators.

| Stage | Completed cells / planned | Actual / planned fits | Status counts |
|---|---:|---:|---|
| S1 | 3601 / 3601 | 3600 / 3600 | {"completed": 3601} |
| S2 | 215 / 292 | 2268 / 3024 | {"completed": 215, "failed_contract": 9, "data_unavailable": 68} |
| S3_synthetic | 1685 / 1717 | 14688 / 19008 | {"completed": 1685, "failed_contract": 4, "data_unavailable": 22, "not_started_prerequisite": 6} |
| S3_measured | 1 / 265 | 0 / 3168 | {"not_started_prerequisite": 264, "completed": 1} |
| S4_synthetic | 1221 / 1221 | 11544 / 14400 | {"completed": 1221} |
| S4_measured | 211 / 385 | 2412 / 4608 | {"completed": 211, "data_unavailable": 48, "failed_contract": 18, "not_started_prerequisite": 108} |

## Gates

```json
{
  "v3_S1_gate": {
    "status": "completed",
    "passed": true,
    "completed": 24,
    "expected": 24,
    "mean_nrmse": {
      "r": 0.4842879153068124,
      "clean_EEG": 0.4842879153068124,
      "clean_HbO": 0.1486210528861541,
      "clean_HbR": 0.16291038947375336
    },
    "mean_r_correlation": 0.8709836882269576,
    "engineering_passed": true,
    "interpretation": "Gaussian mean precheck for bounded feature-missing diagnostics; no Student-t or teacher qualification",
    "rows": [],
    "task_id": "v3_S1_gate",
    "family": "S1",
    "kind": "v3_gate",
    "elapsed_seconds": 0.33379300695378333,
    "peak_rss_bytes": 684650496,
    "completed_at": "2026-09-10T15:44:20.115455+00:00"
  },
  "v3_gain_screen": {
    "status": "completed",
    "passed": false,
    "axis": "gain",
    "conditions": [
      {
        "condition": "matched",
        "complete": true,
        "paired_assessment_counts": [
          6,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            0.0713240120071154,
            0.0713240120071154,
            -0.009726217876680376,
            -0.0101411426346485
          ],
          [
            0.04905573920084618,
            0.04905573920084618,
            0.01085253821957393,
            0.019216290162340288
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0
          ]
        ],
        "mean_nrmse_change": [
          0.030094937801990393,
          0.030094937801990393,
          0.0002815800857233885,
          0.002268786881922947
        ],
        "monte_carlo_se": [
          0.017960046195444244,
          0.017960046195444244,
          0.004203765679059529,
          0.006134051175454923
        ],
        "lower_one_sided95": [
          -0.012171578201719692,
          -0.012171578201719692,
          -0.009611408351850164,
          -0.012166864861595812
        ],
        "upper_one_sided95": [
          0.07236145380570047,
          0.07236145380570047,
          0.01017456852329694,
          0.016704438625441707
        ],
        "verdict": "inconclusive",
        "strict_zero_sensitivity": false
      },
      {
        "condition": "gain_low",
        "complete": true,
        "paired_assessment_counts": [
          6,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            -0.037982332832326375,
            -0.037982332832326375,
            0.0214034569361501,
            0.018925481648823485
          ],
          [
            -0.03950146777329319,
            -0.03950146777329319,
            -0.01133230545470612,
            -0.011465570353693527
          ],
          [
            -0.01930009207465702,
            -0.01930009207465702,
            -0.05753976614718063,
            -0.0653261197614134
          ],
          [
            -0.03372121319750401,
            -0.03372121319750401,
            0.010180052960780304,
            0.00856564334591371
          ]
        ],
        "mean_nrmse_change": [
          -0.03262627646944515,
          -0.03262627646944515,
          -0.009322140426239087,
          -0.012325141280092433
        ],
        "monte_carlo_se": [
          0.004607436671559859,
          0.004607436671559859,
          0.017448443277713048,
          0.018759150873396916
        ],
        "lower_one_sided95": [
          -0.043469249460459145,
          -0.043469249460459145,
          -0.05038466883022265,
          -0.05647224101347542
        ],
        "upper_one_sided95": [
          -0.02178330347843116,
          -0.02178330347843116,
          0.03174038797774448,
          0.031821958453290555
        ],
        "verdict": "inconclusive",
        "strict_zero_sensitivity": false
      },
      {
        "condition": "gain_high",
        "complete": true,
        "paired_assessment_counts": [
          6,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            -0.10630514871718628,
            -0.10630514871718628,
            -0.06173928559015022,
            -0.06801611309859457
          ],
          [
            -0.11306883154388563,
            -0.11306883154388563,
            -0.05091247201870975,
            -0.058599060314765306
          ],
          [
            -0.1364711765850594,
            -0.1364711765850594,
            -0.02993575625183999,
            -0.03508147679401987
          ],
          [
            -0.11688655097758409,
            -0.11688655097758409,
            -0.043330221956692616,
            -0.05628613386708467
          ]
        ],
        "mean_nrmse_change": [
          -0.11818292695592884,
          -0.11818292695592884,
          -0.04647943395434814,
          -0.054495696018616105
        ],
        "monte_carlo_se": [
          0.006476729196045194,
          0.006476729196045194,
          0.006684096215130191,
          0.00695076046411108
        ],
        "lower_one_sided95": [
          -0.133425024623015,
          -0.133425024623015,
          -0.062209541581732794,
          -0.07085336153892127
        ],
        "upper_one_sided95": [
          -0.10294082928884267,
          -0.10294082928884267,
          -0.030749326326963488,
          -0.03813803049831094
        ],
        "verdict": "no_material_degradation_detected",
        "strict_zero_sensitivity": true
      },
      {
        "condition": "g_low",
        "complete": true,
        "paired_assessment_counts": [
          6,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            -0.04965301901084553,
            -0.04965301901084553,
            -0.008503308090567131,
            -0.01263291401494195
          ],
          [
            -0.045315732628393245,
            -0.045315732628393245,
            0.03593581036098292,
            0.03710968542972147
          ],
          [
            -0.050989565664519905,
            -0.050989565664519905,
            0.0053932666586558875,
            0.0022386613008846326
          ],
          [
            -0.0502345545007974,
            -0.0502345545007974,
            0.0009230398113284471,
            0.011775030774862668
          ]
        ],
        "mean_nrmse_change": [
          -0.04904821795113902,
          -0.04904821795113902,
          0.008437202185100032,
          0.009622615872631704
        ],
        "monte_carlo_se": [
          0.001273886976921399,
          0.001273886976921399,
          0.009612849558790531,
          0.010448298419846274
        ],
        "lower_one_sided95": [
          -0.05204613698269608,
          -0.05204613698269608,
          -0.014185326470808444,
          -0.014966027584532186
        ],
        "upper_one_sided95": [
          -0.04605029891958196,
          -0.04605029891958196,
          0.031059730841008507,
          0.034211259329795594
        ],
        "verdict": "inconclusive",
        "strict_zero_sensitivity": false
      },
      {
        "condition": "g_high",
        "complete": false,
        "paired_assessment_counts": [
          0,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            -0.052663847508546136,
            -0.052663847508546136,
            0.021363762766865902,
            0.018588366026433156
          ],
          [
            -0.03977883694268167,
            -0.03977883694268167,
            -0.017726115005750814,
            -0.01479202962700645
          ],
          [
            -0.04200017969394292,
            -0.04200017969394292,
            0.012156655368545002,
            0.04627074445413354
          ]
        ],
        "verdict": "incomplete"
      },
      {
        "condition": "w_low",
        "complete": true,
        "paired_assessment_counts": [
          6,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            0.018097242299078043,
            0.018097242299078043,
            0.009691930466290539,
            0.012934798077542276
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0
          ]
        ],
        "mean_nrmse_change": [
          0.004524310574769511,
          0.004524310574769511,
          0.0024229826165726347,
          0.003233699519385569
        ],
        "monte_carlo_se": [
          0.004524310574769511,
          0.004524310574769511,
          0.0024229826165726347,
          0.0032336995193855693
        ],
        "lower_one_sided95": [
          -0.006123036499580276,
          -0.006123036499580276,
          -0.0032791760764298503,
          -0.00437637068867266
        ],
        "upper_one_sided95": [
          0.015171657649119298,
          0.015171657649119298,
          0.00812514130957512,
          0.010843769727443798
        ],
        "verdict": "no_material_degradation_detected",
        "strict_zero_sensitivity": false
      },
      {
        "condition": "w_high",
        "complete": true,
        "paired_assessment_counts": [
          6,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            -0.021743425201919676,
            -0.021743425201919676,
            0.005717061846185273,
            0.003482781189576458
          ]
        ],
        "mean_nrmse_change": [
          -0.005435856300479919,
          -0.005435856300479919,
          0.0014292654615463182,
          0.0008706952973941145
        ],
        "monte_carlo_se": [
          0.005435856300479919,
          0.005435856300479919,
          0.0014292654615463182,
          0.0008706952973941145
        ],
        "lower_one_sided95": [
          -0.018228401754866473,
          -0.018228401754866473,
          -0.0019343156142819383,
          -0.001178367178347094
        ],
        "upper_one_sided95": [
          0.007356689153906634,
          0.007356689153906634,
          0.0047928465373745744,
          0.002919757773135323
        ],
        "verdict": "no_material_degradation_detected",
        "strict_zero_sensitivity": false
      },
      {
        "condition": "gw_low",
        "complete": true,
        "paired_assessment_counts": [
          6,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            -0.015511327052445123,
            -0.015511327052445123,
            -0.030044965214341007,
            -0.02546652916474344
          ],
          [
            -0.0754431967450455,
            -0.0754431967450455,
            0.05341406368769217,
            0.045926153268378206
          ],
          [
            -0.03510459333170022,
            -0.03510459333170022,
            0.025231870843217024,
            0.027765779841995752
          ],
          [
            -0.027757290981277643,
            -0.027757290981277643,
            -0.04394326848374041,
            -0.030637414205161802
          ]
        ],
        "mean_nrmse_change": [
          -0.03845410202761712,
          -0.03845410202761712,
          0.001164425208206945,
          0.004396997435117179
        ],
        "monte_carlo_se": [
          0.012974992057291659,
          0.012974992057291659,
          0.022945581893879272,
          0.01912679402340343
        ],
        "lower_one_sided95": [
          -0.0689889739020914,
          -0.0689889739020914,
          -0.05283486821109931,
          -0.04061530024454651
        ],
        "upper_one_sided95": [
          -0.007919230153142848,
          -0.007919230153142848,
          0.055163718627513195,
          0.049409295114780864
        ],
        "verdict": "inconclusive",
        "strict_zero_sensitivity": false
      },
      {
        "condition": "gw_high",
        "complete": true,
        "paired_assessment_counts": [
          6,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            -0.123630364682351,
            -0.123630364682351,
            0.10755126964231186,
            0.2206351053086247
          ],
          [
            -0.19175616954437347,
            -0.19175616954437347,
            -0.021634674424631673,
            -0.004561972278700937
          ],
          [
            -0.10683509202008286,
            -0.10683509202008286,
            0.04343805253012314,
            0.06840005666727812
          ]
        ],
        "mean_nrmse_change": [
          -0.10555540656170183,
          -0.10555540656170183,
          0.032338661936950834,
          0.07111829742430047
        ],
        "monte_carlo_se": [
          0.03968716952527659,
          0.03968716952527659,
          0.028488364471676932,
          0.052557888817618165
        ],
        "lower_one_sided95": [
          -0.198953740153269,
          -0.198953740153269,
          -0.03470481332800102,
          -0.05256951632946176
        ],
        "upper_one_sided95": [
          -0.012157072970134672,
          -0.012157072970134672,
          0.0993821372019027,
          0.19480611117806268
        ],
        "verdict": "inconclusive",
        "strict_zero_sensitivity": false
      },
      {
        "condition": "student_t",
        "complete": true,
        "paired_assessment_counts": [
          6,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            0.06301854834455049,
            0.06301854834455049,
            0.05082188922628204,
            0.04470678707554384
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0
          ]
        ],
        "mean_nrmse_change": [
          0.015754637086137623,
          0.015754637086137623,
          0.01270547230657051,
          0.01117669676888596
        ],
        "monte_carlo_se": [
          0.015754637086137623,
          0.015754637086137623,
          0.01270547230657051,
          0.01117669676888596
        ],
        "lower_one_sided95": [
          -0.021321749760951404,
          -0.021321749760951404,
          -0.01719512164159971,
          -0.015126132728877943
        ],
        "upper_one_sided95": [
          0.05283102393322665,
          0.05283102393322665,
          0.042606066254740727,
          0.03747952626664986
        ],
        "verdict": "inconclusive",
        "strict_zero_sensitivity": false
      },
      {
        "condition": "independent_pairing",
        "complete": true,
        "paired_assessment_counts": [
          6,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            -0.11455279249465744,
            -0.11455279249465744,
            -0.05758940308803764,
            -0.06592679505984267
          ],
          [
            -0.1335804605335684,
            -0.1335804605335684,
            -0.08525273559569839,
            -0.09197334315410528
          ],
          [
            -0.22830625125357762,
            -0.22830625125357762,
            -0.02368227700905229,
            0.009457411538954822
          ],
          [
            -0.18497024697160658,
            -0.18497024697160658,
            -0.015170060244353162,
            0.004965916270359125
          ]
        ],
        "mean_nrmse_change": [
          -0.16535243781335252,
          -0.16535243781335252,
          -0.04542361898428537,
          -0.0358692026011585
        ],
        "monte_carlo_se": [
          0.025719810081609402,
          0.025719810081609402,
          0.01613053160990881,
          0.025451166204869754
        ],
        "lower_one_sided95": [
          -0.2258804984094594,
          -0.2258804984094594,
          -0.08338462225895976,
          -0.09576504652076287
        ],
        "upper_one_sided95": [
          -0.10482437721724565,
          -0.10482437721724565,
          -0.00746261570961098,
          0.024026641318445868
        ],
        "verdict": "inconclusive",
        "strict_zero_sensitivity": false
      }
    ],
    "verdict": "incomplete",
    "true_nrmse_order": [
      "r",
      "clean_EEG",
      "clean_HbO",
      "clean_HbR"
    ],
    "margin": 0.02,
    "independent_panels_per_condition": 4,
    "uncertainty": "approximate one-sided paired t bounds, df=3; diagnostic with four displayed panel points",
    "continuation": "incomplete panels block measured adaptation; inconclusive complete panels permit only a mechanism diagnostic",
    "teacher_qualification": "none",
    "rows": [],
    "task_id": "v3_gain_screen",
    "family": "S3_synthetic",
    "kind": "v3_gate",
    "elapsed_seconds": 0.795642445969861,
    "peak_rss_bytes": 1103720448,
    "completed_at": "2026-09-10T16:33:51.956332+00:00"
  },
  "v3_process_screen": {
    "status": "completed",
    "passed": true,
    "axis": "process",
    "conditions": [
      {
        "condition": "q_low",
        "complete": true,
        "paired_assessment_counts": [
          6,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            0.0003322441725600496,
            0.0003322441725600496,
            0.0018722030201700098,
            3.0309001596620182e-05
          ],
          [
            0.010911406830122724,
            0.010911406830122724,
            -0.006824689015251802,
            0.006830288536512282
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0
          ]
        ],
        "mean_nrmse_change": [
          0.0028109127506706936,
          0.0028109127506706936,
          -0.001238121498770448,
          0.0017151493845272256
        ],
        "monte_carlo_se": [
          0.0027013000455817044,
          0.0027013000455817044,
          0.0019137603672865714,
          0.0017050613498887262
        ],
        "lower_one_sided95": [
          -0.0035462280030297885,
          -0.0035462280030297885,
          -0.005741895170115572,
          -0.00229747965039474
        ],
        "upper_one_sided95": [
          0.009168053504371176,
          0.009168053504371176,
          0.0032656521725746767,
          0.005727778419449192
        ],
        "verdict": "no_material_degradation_detected",
        "strict_zero_sensitivity": false
      },
      {
        "condition": "matched",
        "complete": true,
        "paired_assessment_counts": [
          6,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            0.005458747585069877,
            0.005458747585069877,
            0.008804751792187386,
            0.00995234649057261
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            0.0033700225908934824,
            0.0033700225908934824,
            -0.004848364288065401,
            -0.0049506532954199265
          ],
          [
            0.005478599657205476,
            0.005478599657205476,
            0.003230812610571581,
            0.0012077968168107342
          ]
        ],
        "mean_nrmse_change": [
          0.003576842458292209,
          0.003576842458292209,
          0.0017968000286733917,
          0.0015523725029908543
        ],
        "monte_carlo_se": [
          0.001290827376104282,
          0.001290827376104282,
          0.0028658111001989253,
          0.0031007727574266994
        ],
        "lower_one_sided95": [
          0.000539056510727211,
          0.000539056510727211,
          -0.004947495025583944,
          -0.005744872723966764
        ],
        "upper_one_sided95": [
          0.006614628405857207,
          0.006614628405857207,
          0.008541095082930727,
          0.008849617729948472
        ],
        "verdict": "no_material_degradation_detected",
        "strict_zero_sensitivity": false
      },
      {
        "condition": "q_high",
        "complete": true,
        "paired_assessment_counts": [
          6,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            -0.013309448672907584,
            -0.013309448672907584,
            -0.020915659789199525,
            -0.027151696198531028
          ],
          [
            -0.008766067606485609,
            -0.008766067606485609,
            -0.01409655234426409,
            -0.009873702642085197
          ],
          [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          [
            -0.013158960735821928,
            -0.013158960735821928,
            -0.02005559638947503,
            -0.028240197191202437
          ]
        ],
        "mean_nrmse_change": [
          -0.00880861925380378,
          -0.00880861925380378,
          -0.013766952130734662,
          -0.016316399007954667
        ],
        "monte_carlo_se": [
          0.0031195154574496864,
          0.0031195154574496864,
          0.00483294833296992,
          0.006875765662652333
        ],
        "lower_one_sided95": [
          -0.016149972865664955,
          -0.016149972865664955,
          -0.025140636019832498,
          -0.0324975745047066
        ],
        "upper_one_sided95": [
          -0.0014672656419426047,
          -0.0014672656419426047,
          -0.0023932682416368256,
          -0.00013522351120273668
        ],
        "verdict": "no_material_degradation_detected",
        "strict_zero_sensitivity": true
      },
      {
        "condition": "student_t",
        "complete": true,
        "paired_assessment_counts": [
          6,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            0.004446523185816111,
            0.004446523185816111,
            -0.0111826239158322,
            0.004400819325844036
          ],
          [
            0.009588114041568238,
            0.009588114041568238,
            0.003309845786434532,
            0.0172946359487629
          ],
          [
            0.007053455930761778,
            0.007053455930761778,
            -0.0047933209064619856,
            0.016567678079751205
          ],
          [
            -0.0011110568186712873,
            -0.0011110568186712873,
            -0.009380242148611257,
            0.010449386287186502
          ]
        ],
        "mean_nrmse_change": [
          0.00499425908486871,
          0.00499425908486871,
          -0.005511585296117727,
          0.012178129910386162
        ],
        "monte_carlo_se": [
          0.0022898087967931978,
          0.0022898087967931978,
          0.003233428713536979,
          0.003012776703960237
        ],
        "lower_one_sided95": [
          -0.0003944932101919603,
          -0.0003944932101919603,
          -0.013121018199593952,
          0.005087971378063383
        ],
        "upper_one_sided95": [
          0.01038301137992938,
          0.01038301137992938,
          0.0020978476073584985,
          0.01926828844270894
        ],
        "verdict": "no_material_degradation_detected",
        "strict_zero_sensitivity": false
      },
      {
        "condition": "independent_pairing",
        "complete": true,
        "paired_assessment_counts": [
          6,
          6,
          6,
          6
        ],
        "expected_per_panel": 6,
        "panel_differences": [
          [
            -0.12443413522546558,
            -0.12443413522546558,
            0.01992455621465261,
            0.017266215906971787
          ],
          [
            -0.15952356527442327,
            -0.15952356527442327,
            -0.03602880791799182,
            -0.024144041057749883
          ],
          [
            -0.13490834102199878,
            -0.13490834102199878,
            -0.012472157837397572,
            0.028682548865006683
          ],
          [
            -0.11917243768255732,
            -0.11917243768255732,
            0.0010975538343737702,
            0.02700590155775312
          ]
        ],
        "mean_nrmse_change": [
          -0.13450961980111123,
          -0.13450961980111123,
          -0.006869713926590752,
          0.012202656317995428
        ],
        "monte_carlo_se": [
          0.008956380367860561,
          0.008956380367860561,
          0.011772320672343836,
          0.012374183386607009
        ],
        "lower_one_sided95": [
          -0.15558723786701117,
          -0.15558723786701117,
          -0.034574262939646355,
          -0.0169182943995777
        ],
        "upper_one_sided95": [
          -0.11343200173521129,
          -0.11343200173521129,
          0.02083483508646485,
          0.04132360703556856
        ],
        "verdict": "inconclusive",
        "strict_zero_sensitivity": false
      }
    ],
    "verdict": "inconclusive",
    "true_nrmse_order": [
      "r",
      "clean_EEG",
      "clean_HbO",
      "clean_HbR"
    ],
    "margin": 0.02,
    "independent_panels_per_condition": 4,
    "uncertainty": "approximate one-sided paired t bounds, df=3; diagnostic with four displayed panel points",
    "continuation": "incomplete panels block measured adaptation; inconclusive complete panels permit only a mechanism diagnostic",
    "teacher_qualification": "none",
    "rows": [],
    "task_id": "v3_process_screen",
    "family": "S4_synthetic",
    "kind": "v3_gate",
    "elapsed_seconds": 0.5494695259840228,
    "peak_rss_bytes": 719695872,
    "completed_at": "2026-09-10T17:06:22.824977+00:00"
  }
}
```

## Measured comparisons

Only complete 72-trial and (where applicable) 12-selection-fold panels define full risk. Common-subset values are descriptive.

| Rule | Reference | Complete common / 72 | Candidate B | Baseline B | Relative improvement |
|---|---|---:|---:|---:|---:|
| S2/O0 | S2/O0 | 21 | 3.64785 | 3.64785 | 0 |
| S2/O1 | S2/O0 | 2 | 0.831134 | 4.8513 | 0.828678 |
| S2/O2 | S2/O0 | 0 | undefined | undefined | undefined |
| S3_measured/O2 | S2/O2 | 0 | undefined | undefined | undefined |
| S4_measured/O0 | S2/O0 | 10 | 1.99858 | 2.23165 | 0.104438 |
| S4_measured/O2 | S2/O2 | 0 | undefined | undefined | undefined |

Detailed evidence: [solver/truth comparison](S1/solver_comparison.csv), [per-mode denominators](mode_status.csv), [failures](failure_attribution.csv), [full residuals](full_fit_residuals.csv), [process replay](process_replay.csv).

[Independent synthetic baseline / selected / oracle panels](synthetic_adaptation.csv).

Original native-mask results have a different input contract and cannot be used as a direct effect-size baseline. Four-panel synthetic intervals are approximate; incomplete or inconclusive screens do not establish harmless adaptation.

![Combined-processing driver recovery](S1/driver_recovery.png)
