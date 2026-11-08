# DecisionRAG: Hierarchical Non-Generative Decision & Recommendation Engine
> 階層的ナビゲーション ✕ ワンフォワードパス一括決定モデル ✕ 密ベクトルハイブリッド合流を採用した、再利用可能なレコメンド / 階層検索エンジン・ボイラープレート

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2.6+-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev)
[![Package Manager](https://img.shields.io/badge/Package_Manager-uv-blueviolet)](https://github.com/astral-sh/uv)
[![Architecture](https://img.shields.io/badge/Architecture-DecisionRAG-success)](#1-コア設計思想)

---

## 1. コア設計思想（Core Architecture）

従来のRAG（Retrieval-Augmented Generation）やエージェント型探索における**「大量テキストのフラット検索による関連性喪失」**や**「自己回帰LLM呼び出しの連続によるレイテンシ増大・トークン浪費・ハルシネーション」**を克服するため、**DecisionRAG**は以下の4つの柱で設計されています。

```text
                           [ ユーザー意図 / クエリ ]
                                      │
                                      ▼
                   Level 1: カテゴリ群（共有文脈一括判定）
                      ┌───────────────┼───────────────┐
                     SF           サスペンス        アニメ
                     ✂️ (Pruned)     ✅ (Survives)    ✂️ (Pruned)
                                      │
                                      ▼
                   Level 2: サブカテゴリ群（監督・時代一括判定）
                     ┌────────────────┴────────────────┐
             ノーラン（初期）                   フィンチャー
               ✅ (Survives)                     ✂️ (Pruned)
                     │
                     ▼
             Level 3: 作品ノード展開
                     ┌────────────────┴────────────────┐
              メメント (2000)                   プレステージ (2006)
               ✅ (Survives)                     ✂️ (Pruned)
                     │
                     ▼
           [ 階層枝刈り通過候補 ]             [ Dense Retrieval (補助候補) ]
                     │                                   │
                     └─────────────────┬─────────────────┘
                                       ▼
                       【ハイブリッド合流 & 重複排除】
                                       ▼
                     【リストワイズ（Listwise）最終順位決定】
                     （全候補を共有文脈で相対比較しスコア較正）
                                       ▼
                                 [ 最終推薦結果 ]
```

### 4つのコア柱
1. **非生成型の決定（Decision）モデル**:
   - 自己回帰（テキスト生成）を行わず、共有文脈（In-context comparison）にある複数候補の適合度/スコアを**1回の計算（ワンフォワードパス）で一括判定**。
2. **階層ナビゲーション ＆ 枝刈り（Pruning）**:
   - `Corpus ➔ Category ➔ Subcategory ➔ Item ➔ Detail (Passage)` のツリー構造を上から順に判定し、見込みのない枝を早めに切り捨てることで探索空間を $O(\log N)$ に圧縮。
3. **ハイブリッド合流（Dense Candidates Rescue）**:
   - 階層探索による枝刈りの取りこぼしを防ぐため、従来の埋め込みベクトル検索（Dense Retrieval）の上位候補を最終判定の直前にマージ・合流。
4. **リストワイズ（Listwise）最終順位決定**:
   - 最終段では、残った候補を並べて相対比較（共有文脈）し、1位から順位を決定。

---

## 2. ディレクトリ構成

他プロジェクトで容易に複製・拡張できるクリーンアーキテクチャ構成です。

```text
.
├── README.md                     # アーキテクチャ解説・クイックスタートガイド
├── pyproject.toml                # uv / pip 設定ファイル
├── .env.example                  # 環境変数テンプレート
├── demo.py                       # CLIによる即時デモ実行スクリプト
├── data/
│   ├── sample_catalog.json       # デモ映画カタログ（デフォルト）
│   ├── sample_books_ochiai.json  # サンプル書籍データセット（落合陽一著作コレクション）
│   ├── sample_ecommerce.json     # サンプルECガジェットデータセット
│   └── datasets/                 # 動的マウントされたカスタムデータセット保管先
├── src/
│   ├── __init__.py
│   ├── config.py                 # Pydantic-Settings によるハイパーパラメータ管理
│   ├── cli_json.py               # JSON 出力 CLI ブリッジ
│   ├── core/
│   │   ├── __init__.py
│   │   ├── tree.py               # 任意階層ツリー構造（動的マウント・アンマウント・トグル対応）
│   │   ├── decision_model.py     # 決定モデル基底クラス (Mock / Cross-Encoder / LLM)
│   │   ├── dense_retriever.py    # 密ベクトル検索（動的再インデックス対応）
│   │   └── engine.py             # 階層枝刈り + 密候補合流 + リストワイズ順位決定
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── recommendation.py     # Pydantic v2 スキーマ & DatasetInputSchema
│   └── api/
│       ├── __init__.py
│       └── app.py                # FastAPI REST API (/recommend, /datasets, /tree)
└── tests/
    ├── __init__.py
    ├── test_tree.py              # ツリーインデックス & マウント/アンマウントのテスト
    └── test_engine.py            # DecisionEngine の探索・動的マウント検索テスト

---

## 3. データセット動的追加・削除（Dynamic Dataset CRUD）

DecisionRAG はハードコードされたデータだけでなく、画面（UI）や API からいつでも任意の階層データセット（書籍、EC商品、社内ドキュメント等）を動的にマウント・アンマウントできます。

### 推奨フォーマット：階層ツリー型 JSON
```json
{
  "id": "books_ochiai",
  "title": "書籍・学術：落合陽一著作コレクション",
  "description": "落合陽一による計算機自然、デジタルネイチャー、魔法の世紀などのメディア論・現代思想書籍群",
  "children": [
    {
      "id": "sub_media_thought",
      "title": "メディア論・デジタル思想",
      "description": "デジタルネイチャー、計算機自然、情報社会論、脱人間中心主義",
      "children": [
        {
          "id": "book_digital_nature",
          "title": "デジタルネイチャー (Digital Nature)",
          "description": "落合陽一著。計算機によって再構築される新しい自然観と脱人間中心主義。",
          "metadata": {
            "author": "落合陽一",
            "year": 2018,
            "tags": ["思想", "AI", "メディアアート"]
          }
        }
      ]
    }
  ]
}
```

### マウント時の自動処理
1. 投入された JSON をパースし、ルートノード（Level 0）の `children` に新規カテゴリ（Level 1）として結合。
2. 最下層（葉ノード）のテキストを自動抽出し、埋め込みベクトルインデックス（Dense Retriever）にも自動同期登録。
3. 削除時は、ツリーおよび埋め込みインデックスの両方から即座に除外。
```

---

## 3. クイックスタート

### 前提条件
- Python 3.10+ (推奨: Python 3.11+)
- パッケージマネージャー `uv` (または標準 `pip`)

### A. `uv` を使用する場合（推奨）
```bash
# 1. 依存関係の同期（仮想環境の作成）
uv sync

# 2. CLIデモの実行（「ノーラン監督の2000年の映画」でメメントを特定）
uv run python demo.py --query "ノーラン監督の2000年の映画"

# 3. テストの実行
uv run pytest

# 4. FastAPIサーバーの起動
uv run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### B. 標準 `pip` / `python` を使用する場合
```bash
# 依存ライブラリのインストール
pip install -e .

# CLIデモの実行
python3 demo.py --query "ノーラン監督の2000年の映画"

# 単体テストの実行
python3 -m unittest discover tests
```

---

## 4. 実行例（CLI Demonstration）

```bash
$ python3 demo.py --query "ノーラン監督の2000年の映画"
```

**出力結果:**
```text
======================================================================
🌲 DecisionRAG Hierarchical Decision & Recommendation Engine
======================================================================
🔍 Query: "ノーラン監督の2000年の映画"
⚙️  Settings: branch_top_k=2, prune_threshold=0.2, dense_top_n=3
----------------------------------------------------------------------
📍 EXPLORATION & DECISION TRACE (階層枝刈り & 決定プロセス):

[Step 1] Level 1 Hierarchical Navigation & Batch Pruning
  ℹ️  Evaluated 3 candidate branches in a single shared forward pass. Preserved top 2 branches, pruned 1.
     ✅ [SURVIVED] サスペンス・ミステリー・クライム      (Score: 0.422) -> Score 0.42 >= 0.2 (Top 1)
     ✅ [SURVIVED] アニメーション・ファンタジー・精神世界   (Score: 0.255) -> Score 0.26 >= 0.2 (Top 2)
     ✂️  [PRUNED] SF・サイエンスフィクション        (Score: 0.087) -> Pruned: Below threshold 0.2

[Step 2] Level 2 Hierarchical Navigation & Batch Pruning
  ℹ️  Evaluated 4 candidate branches in a single shared forward pass. Preserved top 1 branches, pruned 3.
     ✅ [SURVIVED] クリストファー・ノーラン（初期サスペンス・記憶と偽り） (Score: 0.807) -> Score 0.81 >= 0.2 (Top 1)
     ✂️  [PRUNED] 今敏（虚構と現実の越境・心理サスペンス）          (Score: 0.197) -> Pruned: Below threshold 0.2
     ✂️  [PRUNED] デヴィッド・フィンチャー（サイコスリラー）        (Score: 0.136) -> Pruned: Below threshold 0.2
     ✂️  [PRUNED] 宮崎駿（自然・神話・成長ファンタジー）           (Score: 0.105) -> Pruned: Below threshold 0.2

[Step 3] Level 3 Hierarchical Navigation & Batch Pruning
     ✅ [SURVIVED] メメント (Memento)                 (Score: 1.000) -> Score 1.00 >= 0.2 (Top 1)
     ✅ [SURVIVED] プレステージ (The Prestige)          (Score: 0.263) -> Score 0.26 >= 0.2 (Top 2)

[Step 4] Level 4 Hierarchical Navigation & Batch Pruning
     ✅ [SURVIVED] メメント：詳細あらすじと核心テーマ       (Score: 0.912) -> Score 0.91 >= 0.2 (Top 1)
     ✂️  [PRUNED] プレステージ：詳細あらすじと核心テーマ     (Score: 0.070) -> Pruned: Below threshold 0.2

[Step 5] Dense Candidates Rescue (Hybrid Fallback)
  ℹ️  Retrieved Top-3 dense candidates via vector embedding space. Rescued 2 candidates that were bypassed or pruned.
     ✅ [SURVIVED] メメント (Memento)                   (Score: 0.321) -> Corroborated by Dense Retrieval
     🛟 [RESCUED] ブレードランナー 2049 (Blade Runner 2049) (Score: 0.094) -> Rescued via Dense Retrieval
     🛟 [RESCUED] インターステラー (Interstellar)          (Score: 0.059) -> Rescued via Dense Retrieval

======================================================================
🏆 FINAL LISTWISE RANKED RESULTS (最終順位決定 / 2.67ms):
======================================================================
#1  メメント (Memento)  [Score: 1.000 | 🌿 Tree Pruning]
    Path: サスペンス・ミステリー・クライム ➔ クリストファー・ノーラン（初期サスペンス・記憶と偽り） ➔ メメント (Memento)
    Meta: year: 2000, director: Christopher Nolan, runtime_min: 113
    Why:  公開年(2000)一致、監督(Christopher Nolan)一致、クエリ概念との高いコンテキスト類似度
    Desc: 記憶が10分しか保てない男が妻の殺害犯を追う、全編を逆時系列とカラー/白黒で紡ぐ緊迫のサスペンス

#2  インターステラー (Interstellar)  [Score: 0.441 | 🛟 Dense Rescue]
    Path: SF・サイエンスフィクション ➔ クリストファー・ノーラン（SF・時間軸構造） ➔ インターステラー (Interstellar)
    Meta: year: 2014, director: Christopher Nolan, runtime_min: 169
    Why:  監督(Christopher Nolan)一致、階層カテゴリおよび関連メタデータが適合
    Desc: 滅びゆく地球から人類を救うため、ワームホールを越えて新たな居住可能惑星を探す宇宙探査映画
```

---

## 5. API仕様（FastAPI）

サーバーを起動後、ブラウザで `http://localhost:8000/docs` にアクセスするとSwagger UIが利用可能です。

### 主要エンドポイント
- **`POST /recommend`**: ハイパーパラメータ指定および全探索ログ（監査トレース）付きレコメンド
```json
// POST /recommend
{
  "query": "夢の中に潜入する産業スパイ",
  "top_k": 3,
  "branch_top_k": 2,
  "prune_threshold": 0.25,
  "enable_dense_hybrid": true,
  "dense_top_n": 3
}
```
- **`POST /search?query=...`**: 簡易検索エンドポイント
- **`GET /tree`**: フロントエンド可視化用のカタログ木全体構造
- **`GET /health`**: ヘルスチェック & インデックスノード数

---

## 6. 拡張・カスタマイズガイド

### A. 決定モデルをクロスエンコーダー（HuggingFace等）に差し替える
`src/core/decision_model.py` の `BaseDecisionModel` を継承し、クロスエンコーダーのスコアリングを実装します：
```python
from sentence_transformers import CrossEncoder
from src.core.decision_model import BaseDecisionModel
from src.core.tree import TreeNode

class CrossEncoderDecisionModel(BaseDecisionModel):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def judge_relevance_batch(self, query: str, candidates: list[TreeNode]) -> list[float]:
        pairs = [[query, c.get_context_text()] for c in candidates]
        scores = self.model.predict(pairs)
        # 0.0 - 1.0 にシグモイド等で正規化
        return [float(1 / (1 + math.exp(-s))) for s in scores]

    def rerank_listwise(self, query: str, candidates: list[TreeNode]) -> list[tuple[TreeNode, float, str]]:
        scores = self.judge_relevance_batch(query, candidates)
        paired = [(node, score, "クロスエンコーダー判定") for node, score in zip(candidates, scores)]
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired
```

### B. レコメンド用「ユーザー履歴・プロファイル」の注入
クエリ文字列にユーザーの閲覧履歴や購買傾向を埋め込むだけで、ツリー探索時にユーザー文脈が共有されます：
```python
user_profile = "ユーザー嗜好: 2000年代以降のハードSF、クリストファー・ノーラン監督作品、高評価映画"
request = RecommendationRequest(
    query=f"{user_profile} | 探している作品: 時間の逆行を扱ったアクション"
)
```

---

## 7. ライセンス
Apache License 2.0
