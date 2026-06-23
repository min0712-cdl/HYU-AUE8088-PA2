def clean_cell($prefix):
  .id = ($prefix + "-" + .id)
  | if .cell_type == "code" then
      .execution_count = null | .outputs = []
    else . end;

def selected_cells($doc; $prefix; $excluded):
  [
    $doc.cells[]
    | . as $cell
    | select(($excluded | index($cell.id)) == null)
    | clean_cell($prefix)
  ];

def code_cell($id; $source):
  {
    cell_type: "code",
    execution_count: null,
    id: $id,
    metadata: {},
    outputs: [],
    source: $source
  };

def markdown_cell($id; $source):
  {
    cell_type: "markdown",
    id: $id,
    metadata: {},
    source: $source
  };

def cleanup_cell($id; $names):
  code_cell($id; [
    "# 다음 Level을 위해 CPU/GPU 메모리를 정리합니다.\n",
    "import gc\n",
    ("for _name in " + ($names | @json) + ":\n"),
    "    globals().pop(_name, None)\n",
    "gc.collect()\n",
    "if torch.cuda.is_available():\n",
    "    torch.cuda.empty_cache()\n"
  ]);

.[0] as $l1
| .[1] as $l2
| .[2] as $l3
| .[3] as $l4
| .[4] as $l5
| selected_cells($l1; "l1"; ["7bc15fbc", "53aae459"]) as $l1cells
| selected_cells($l2; "l2"; ["5f25c32d"]) as $l2cells
| selected_cells($l3; "l3"; ["b9f7b0c7"]) as $l3cells
| selected_cells($l4; "l4"; ["d990be98"]) as $l4cells
| selected_cells($l5; "l5"; ["13a75f8f", "submit"]) as $l5cells
| ($l1cells | map(if .id == "l1-setup" then .source = .source[3:] else . end)) as $l1cells
| {
    cells: (
      [
        markdown_cell("submission-title"; [
          "# PA2 통합 제출 노트북\n",
          "\n",
          "Level 1~5를 순서대로 실행합니다. Kaggle 제출 단계는 별도 공지에 따라 제외했습니다.\n"
        ]),
        code_cell("shared-setup"; [
          "import os\n",
          "import sys\n",
          "\n",
          "# 제출 기본값: 모든 모델을 처음부터 학습. 기존 checkpoint 평가만 할 때 False.\n",
          "RUN_ALL_TRAINING = True\n",
          "ENABLE_WANDB = False\n",
          "os.environ[\"AUE8088_RUN_TRAINING\"] = \"1\" if RUN_ALL_TRAINING else \"0\"\n",
          "os.environ[\"AUE8088_USE_WANDB\"] = \"1\" if ENABLE_WANDB else \"0\"\n",
          "\n",
          "repo_url = \"https://github.com/min0712-cdl/HYU-AUE8088-PA2.git\"\n",
          "repo_name = \"HYU-AUE8088-PA2\"\n",
          "if not os.path.exists(f\"/content/{repo_name}\"):\n",
          "    !git clone {repo_url}\n",
          "else:\n",
          "    !git -C /content/{repo_name} pull\n",
          "%cd /content/{repo_name}\n",
          "\n",
          "from google.colab import drive\n",
          "drive.mount(\"/content/drive\")\n",
          "ARTIFACT_ROOT = os.environ.get(\"AUE8088_ARTIFACT_ROOT\", \"/content/drive/MyDrive/AUE8088_PA2\")\n",
          "CHECKPOINT_DIR = os.path.join(ARTIFACT_ROOT, \"checkpoints\")\n",
          "OUTPUT_DIR = os.path.join(ARTIFACT_ROOT, \"outputs\")\n",
          "os.makedirs(CHECKPOINT_DIR, exist_ok=True)\n",
          "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
          "\n",
          "%load_ext autoreload\n",
          "%autoreload 2\n",
          "!pip install -q -r requirements.txt\n"
        ])
      ]
      + $l1cells
      + [cleanup_cell("cleanup-l1"; ["vgg_model", "r18_model", "r50_model", "r18_w211_model", "train_loader", "val_loader", "train_ds", "val_ds"])]
      + $l2cells
      + [cleanup_cell("cleanup-l2"; ["scratch_model", "pretrained_model", "train_loader", "val_loader", "train_ds", "val_ds"])]
      + $l3cells
      + [cleanup_cell("cleanup-l3"; ["results", "best_result", "train_loader", "val_loader", "train_ds", "val_ds"])]
      + $l4cells
      + [cleanup_cell("cleanup-l4"; ["model", "val_loader", "val_ds", "preds", "probs", "targets"])]
      + $l5cells
      + [
        code_cell("artifact-check"; [
          "# 제출 산출물이 모두 생성되었는지 확인합니다.\n",
          "required_checkpoints = [\n",
          "    \"level1_vgg16.pth\", \"level1_resnet18.pth\", \"level1_resnet50.pth\",\n",
          "    \"level1_resnet18-w211.pth\", \"level2_vit_s16_scratch.pth\",\n",
          "    \"level2_vit_s16_pretrained.pth\", \"level3_best.pth\",\n",
          "    \"level5_hybrid-rare-hard-joint-1000.pth\", \"level5_random-1000.pth\",\n",
          "    \"level5_final.pth\",\n",
          "]\n",
          "missing = [name for name in required_checkpoints if not os.path.exists(os.path.join(CHECKPOINT_DIR, name))]\n",
          "picks_path = os.path.join(OUTPUT_DIR, \"level5_picks.json\")\n",
          "assert not missing, f\"Missing checkpoints: {missing}\"\n",
          "assert os.path.exists(picks_path), f\"Missing artifact: {picks_path}\"\n",
          "print(f\"Verified {len(required_checkpoints)} checkpoints\")\n",
          "print(f\"Curation artifact: {picks_path}\")\n"
        ])
      ]
    ),
    metadata: {
      kernelspec: {
        display_name: "Python 3",
        language: "python",
        name: "python3"
      },
      language_info: {name: "python"}
    },
    nbformat: 4,
    nbformat_minor: 5
  }
