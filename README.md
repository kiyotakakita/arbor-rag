# 🌲 Arbor-RAG (アーバーラグ)
### Self-Growing, Generation-Free Hierarchical Decision & Recommendation Engine
> **テキスト生成の遅延を排した「階層型決定探索」 ✕ 知識の空白から自律成長する「Gemini 庭師エージェント」**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)]()
[![Gemini](https://img.shields.io/badge/Powered%20by-Gemini%202.5%20Flash-orange.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)]()

---

## 💡 概要（About）

**Arbor-RAG** は、従来の RAG や生成型レコメンドが抱える**「自己回帰テキスト生成による致命的な遅延」「フラット検索によるハルシネーション」**を解決するために生まれた、次世代の意思決定・推薦エンジンです。

1. **生成しない決定モデル（Generation-Free）**:
   文章を長文生成するのではなく、階層ツリーを上からワンフォワードパス（または一括判定）で枝刈り（Pruning）しながら数ミリ秒で駆け下ります。
2. **自己進化・自律接ぎ木（Self-Growing Knowledge Tree）**:
   未登録の概念や未知のクエリが与えられ、適合スコアが閾値を下回った場合、**「🌱 庭師エージェント（Tree Gardener）」**が自律発火。Gemini 2.5 Flash を通じて実在する専門知識ツリーを即座に自動生成し、既存カタログに動的接ぎ木（Auto-Mount）して再探索します。**使えば使うほど、リポジトリが勝手に育ちます。**
3. **埋め込みハイブリッド合流（Dense Rescue）**:
   階層枝刈りの取りこぼしを防ぐため、密ベクトル検索（Dense Retrieval）の上位候補を最終段でマージ・合流させます。

---

## 🚀 奇跡の自己進化デモ（Real-world Trace）

リポジトリ内に「映画カタログ」しかない状態で、全く未知のクエリ **「枕草子の作者」** を投げたときの実際の実行ログです。

```text
🔍 Query: "枕草子の作者"
⚙️  Settings: branch_top_k=2, prune_threshold=0.2, dense_top_n=3, auto_grow=True
----------------------------------------------------------------------
🌱 庭師エージェントが発火: Gemini に問い合わせ中...
HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent "HTTP/1.1 200 OK"

✅ 新規ツリー接ぎ木完了: 
   【枕草子の作者 ➔ 清少納言 ➔ 講談社学術文庫(橋本治訳), Penguin Classics, 新潮日本古典集成】

📍 EXPLORATION & DECISION TRACE:
[Step 1] 🌱 [GARDENER AUTO-MOUNT]
  ℹ️ 【進化前】パーフェクトブルー (Score: 0.376) 
  ➔ 【自己進化後】枕草子 全現代語訳 (講談社学術文庫) (Score: 0.606)

[Step 2] Level 1 Hierarchical Navigation & Batch Pruning
     ✅ [SURVIVED] 枕草子の作者                           (Score: 0.720) 
     ✂️  [PRUNED] アニメーション・ファンタジー・精神世界              (Score: 0.106) -> 一撃枝刈り！
     ✂️  [PRUNED] SF・サイエンスフィクション                   (Score: 0.038)
     ✂️  [PRUNED] サスペンス・ミステリー・クライム                 (Score: 0.038)

[Step 3] Level 2 Hierarchical Navigation & Batch Pruning
     ✅ [SURVIVED] 清少納言                             (Score: 0.600)

[Step 4] Level 3 Hierarchical Navigation & Batch Pruning
     ✅ [SURVIVED] 枕草子 全現代語訳 (講談社学術文庫)              (Score: 0.606)
     ✅ [SURVIVED] The Pillow Book (Penguin Classics) (Score: 0.517)

======================================================================
🏆 FINAL LISTWISE RANKED RESULTS:
======================================================================
#1  枕草子 全現代語訳 (講談社学術文庫)  [Score: 0.606 | 🌿 Tree Pruning]
    Path: 枕草子の作者 ➔ 清少納言 ➔ 枕草子 全現代語訳 (講談社学術文庫)
    Meta: translator: 橋本治, publisher: 講談社学術文庫, year: 2007
    Desc: 橋本治による現代語訳。原文の魅力を損なわずに親しみやすい言葉で翻訳されている。

#2  The Pillow Book (Penguin Classics)  [Score: 0.417 | 🌿 Tree Pruning]
    Path: 枕草子の作者 ➔ 清少納言 ➔ The Pillow Book (Penguin Classics)
    Meta: translator: Meredith McKinney, publisher: Penguin Classics, year: 2006
    Desc: メレディス・マッキニーによる枕草子の英訳版。詳細な解説が含まれる。
```

> **進化前は映画しかなく『パーフェクトブルー (0.37)』を返していた世界が、クエリを1回投げただけで、実在する書誌データ（清少納言、橋本治訳）を自力で学習・接ぎ木し、映画階層を 0.10 で叩き落とすまでに自己進化しました。**

---

## 🏗️ アーキテクチャ

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
                   Level 2: サブカテゴリ群（一括判定）
                                      │
                                      ▼
                   Level 3: アイテム・作品ノード展開
                                      │
                                      ▼
           [ 階層枝刈り通過候補 ]             [ Dense Retrieval (補助候補) ]
                     │                                   │
                     └─────────────────┬─────────────────┘
                                       ▼
                       【ハイブリッド合流 & 重複排除】
                                       ▼
                     【リストワイズ（Listwise）最終順位決定】
                                       │
                      ┌────────────────┴────────────────┐
             (Topスコア >= 0.50)               (Topスコア < 0.50)
                      ▼                                 ▼
               [ 最終推薦結果 ]              🚨【知識の空白を検知！】
                                                        │
                                                        ▼
                                         🌱【Tree Gardener (Gemini) 発火】
                                         実在知識ツリーを生成 ➔ 動的接ぎ木
                                                        │
                                                        ▼
                                                【自動で再探索・即答】
```

---

## 🛠️ クイックスタート

### 前提条件
- Python 3.10+
- Google Gemini API Key（`GEMINI_API_KEY`）

### 1. インストール
```bash
git clone https://github.com/kiyotakakita/arbor-rag.git
cd arbor-rag

# 仮想環境の作成と有効化
python3 -m venv .venv
source .venv/bin/activate

# 依存ライブラリのインストール
pip install -r requirements.txt
pip install google-genai
```

### 2. 環境変数の設定
```bash
export GEMINI_API_KEY="AIzaSyあなたのAPIキー"
```

### 3. CLI による自己進化デモの実行
```bash
# 既存データ（映画）にあるクエリ ➔ 超高速ミリ秒で特定
python demo.py --query "ノーラン監督の2000年の映画"

# 既存データにない未知のクエリ ➔ 庭師が発火して自動接ぎ木・自己進化！
python demo.py --query "枕草子の作者" --auto-grow
python demo.py --query "富野由悠季の代表作" --auto-grow
```

### 4. Web UI（シミュレーター画面）の起動
```bash
# 依存パッケージのインストール
npm install --legacy-peer-deps

# 開発サーバー起動
npm run dev
```

---

## 📁 ディレクトリ構成

```text
.
├── demo.py                       # CLI デモ実行スクリプト（自己進化対応）
├── data/
│   ├── sample_catalog.json       # コアカタログ（映画ツリー）
│   └── datasets/                 # 動的マウントされたデータセット保管先
├── src/
│   ├── core/
│   │   ├── tree.py               # 任意階層ツリー構造（動的マウント/アンマウント対応）
│   │   ├── gardener.py           # 🌱 Gemini 2.5 Flash による自律接ぎ木エージェント
│   │   ├── decision_model.py     # 決定モデル基底クラス
│   │   ├── dense_retriever.py    # 密ベクトル検索（動的再インデックス対応）
│   │   └── engine.py             # 階層枝刈り + 密候補合流 + 自己進化ループ
│   └── api/
│       └── app.py                # FastAPI REST API (/recommend, /datasets, /tree)
└── tests/
```

---

## 📜 ライセンス
Apache License 2.0
```

---
