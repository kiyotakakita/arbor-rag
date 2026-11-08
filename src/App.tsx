import React, { useState, useEffect } from "react";
import {
  GitFork,
  Cpu,
  Layers,
  Search,
  Sparkles,
  Sliders,
  CheckCircle2,
  XCircle,
  LifeBuoy,
  Terminal,
  FileCode,
  FolderTree,
  BookOpen,
  Copy,
  Check,
  Play,
  ShieldAlert,
  Flame,
  Film,
  Zap,
  RefreshCw,
  Code2,
  Database,
  Trash2,
  PlusCircle,
  ToggleLeft,
  ToggleRight,
  RotateCcw,
  Sprout,
  Scissors,
  ArrowRight,
  Clock,
  History,
  TrendingUp,
  AlertTriangle,
} from "lucide-react";

interface EvaluatedNode {
  id: string;
  title: string;
  level: number;
  score: number;
  survived: boolean;
  action: "survived" | "pruned" | "rescued";
  reason?: string;
}

interface DecisionTraceStep {
  stage: string;
  level: number;
  description: string;
  evaluated: EvaluatedNode[];
  survivors_count: number;
  pruned_count: number;
}

interface CandidateResult {
  rank: number;
  id: string;
  title: string;
  description: string;
  level: number;
  score: number;
  source: "tree_pruning" | "dense_rescue" | "both";
  reasoning: string;
  breadcrumb: string[];
  metadata: Record<string, any>;
}

interface RecommendationResponse {
  query: string;
  results: CandidateResult[];
  total_candidates_evaluated: number;
  tree_survivors_count: number;
  dense_rescued_count: number;
  execution_time_ms: number;
  trace: DecisionTraceStep[];
}

interface TreeNodeData {
  id: string;
  title: string;
  description: string;
  level: number;
  metadata?: Record<string, any>;
  children?: TreeNodeData[];
  enabled?: boolean;
}

interface MountedDataset {
  id: string;
  title: string;
  description: string;
  is_default: boolean;
  enabled: boolean;
  node_count: number;
  leaf_count: number;
}

interface DatasetPreset {
  preset_id: string;
  label: string;
  description: string;
  data: any;
}

interface GardenerEvent {
  event_id: string;
  timestamp: string;
  query: string;
  top_score_before: number;
  top_title_before: string;
  gap_detected: boolean;
  missing_concept: string;
  grafted_dataset_id?: string;
  grafted_title?: string;
  grafted_nodes_count: number;
  top_score_after?: number;
  top_title_after?: string;
  accuracy_gain_pct: number;
  status: string;
  message: string;
}

const PRESET_QUERIES = [
  {
    label: "落合陽一のデジタルネイチャーと計算機自然",
    query: "落合陽一のデジタルネイチャーと計算機自然",
    highlight: "【未学習時は低スコア】庭師エージェントが発火して自動接ぎ木 ➔ 100%へ自己進化！",
    tag: "自己進化テスト",
  },
  {
    label: "ノーラン監督の2000年の映画",
    query: "ノーラン監督の2000年の映画",
    highlight: "【映画カタログ】『メメント』の特定（公開年2000年 + 監督ノーランで最高スコア）",
    tag: "映画",
  },
  {
    label: "Sonyのノイズキャンセリングヘッドホン",
    query: "Sonyのノイズキャンセリングヘッドホン WH-1000XM5 ハイレゾ",
    highlight: "【ECカタログ】未登録時はギャップ検知 ➔ ガジェットツリー自動生成",
    tag: "自己進化テスト",
  },
  {
    label: "夢の中に潜入する産業スパイ",
    query: "他人の夢の中に潜入し潜在意識の深層でアイデアを植え付ける産業スパイ",
    highlight: "【映画カタログ】『インセプション』の特定（SF階層 ➔ ノーラン ➔ 多重夢）",
    tag: "映画",
  },
  {
    label: "七つの大罪を見立てた猟奇殺人事件",
    query: "キリスト教の七つの大罪を模した雨の降る街の連続猟奇殺人犯とベテラン刑事",
    highlight: "【映画カタログ】『セブン』の特定（サスペンス ➔ デヴィッド・フィンチャー）",
    tag: "映画",
  },
  {
    label: "アイドルから女優へ転身、夢と虚構の崩壊",
    query: "アイドルグループを脱退して女優へ、ストーカーと未麻の部屋、精神的境界の崩壊",
    highlight: "【映画カタログ】『パーフェクトブルー』の特定（アニメ ➔ 今敏）",
    tag: "映画",
  },
];

const EMPTY_DATASET_TEMPLATE = {
  id: "custom_my_catalog",
  title: "新規カテゴリ・カタログ名",
  description: "決定モデルが判定に使用する説明文（充実させるほど精度向上）",
  children: [
    {
      id: "sub_category_1",
      title: "サブカテゴリ名",
      description: "サブカテゴリのテーマや対象領域の説明",
      children: [
        {
          id: "item_example_1",
          title: "推奨対象の作品・商品・ドキュメント名",
          description: "アイテムの詳細な要約や紹介文。検索クエリと照合されます。",
          metadata: {
            author: "著者名またはブランド",
            year: 2026,
            tags: ["タグ1", "タグ2"],
          },
        },
      ],
    },
  ],
};

export default function App() {
  const [activeTab, setActiveTab] = useState<"workbench" | "gardener" | "datasets" | "tree" | "code" | "architecture">("workbench");

  // Query & parameters state
  const [query, setQuery] = useState("落合陽一のデジタルネイチャーと計算機自然");
  const [branchTopK, setBranchTopK] = useState(2);
  const [pruneThreshold, setPruneThreshold] = useState(0.2);
  const [enableDenseHybrid, setEnableDenseHybrid] = useState(true);
  const [denseTopN, setDenseTopN] = useState(3);
  const [topK, setTopK] = useState(3);

  // Auto-Gardener settings
  const [autoGrowMode, setAutoGrowMode] = useState<boolean>(true);
  const [gardenerRunning, setGardenerRunning] = useState<boolean>(false);
  const [gardenerEvents, setGardenerEvents] = useState<GardenerEvent[]>([]);
  const [gardenerBanner, setGardenerBanner] = useState<string | null>(null);

  // Results state
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<RecommendationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Tree & files state
  const [treeData, setTreeData] = useState<TreeNodeData | null>(null);
  const [filesMap, setFilesMap] = useState<Record<string, string>>({});
  const [selectedFile, setSelectedFile] = useState<string>("src/core/gardener.py");
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  // Dataset Management state
  const [datasets, setDatasets] = useState<MountedDataset[]>([]);
  const [presets, setPresets] = useState<DatasetPreset[]>([]);
  const [jsonInput, setJsonInput] = useState<string>("");
  const [mountLoading, setMountLoading] = useState(false);
  const [mountError, setMountError] = useState<string | null>(null);
  const [mountSuccess, setMountSuccess] = useState<string | null>(null);

  // Initial load
  useEffect(() => {
    fetchTree();
    fetchDatasets();
    fetchPresets();
    fetchFiles();
    fetchGardenerEvents();
    handleSearch("落合陽一のデジタルネイチャーと計算機自然");
  }, []);

  const fetchTree = async () => {
    try {
      const res = await fetch("/api/tree");
      if (res.ok) {
        const data = await res.json();
        setTreeData(data);
      }
    } catch (e) {
      console.error("Failed to load tree", e);
    }
  };

  const fetchDatasets = async () => {
    try {
      const res = await fetch("/api/datasets");
      if (res.ok) {
        const data = await res.json();
        setDatasets(data.datasets || []);
      }
    } catch (e) {
      console.error("Failed to load datasets", e);
    }
  };

  const fetchGardenerEvents = async () => {
    try {
      const res = await fetch("/api/gardener/events");
      if (res.ok) {
        const data = await res.json();
        setGardenerEvents(data.events || []);
      }
    } catch (e) {
      console.error("Failed to load gardener events", e);
    }
  };

  const fetchPresets = async () => {
    try {
      const res = await fetch("/api/datasets/presets");
      if (res.ok) {
        const data = await res.json();
        setPresets(data.presets || []);
        if (data.presets && data.presets.length > 0 && !jsonInput) {
          setJsonInput(JSON.stringify(data.presets[0].data, null, 2));
        }
      }
    } catch (e) {
      console.error("Failed to load presets", e);
    }
  };

  const fetchFiles = async () => {
    try {
      const res = await fetch("/api/files");
      if (res.ok) {
        const data = await res.json();
        setFilesMap(data.files || {});
      }
    } catch (e) {
      console.error("Failed to load files", e);
    }
  };

  const handleSearch = async (targetQuery?: string) => {
    const q = targetQuery !== undefined ? targetQuery : query;
    if (!q.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          top_k: topK,
          branch_top_k: branchTopK,
          prune_threshold: pruneThreshold,
          enable_dense_hybrid: enableDenseHybrid,
          dense_top_n: denseTopN,
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || "Failed to execute search");
      }

      const data: RecommendationResponse = await res.json();
      setResponse(data);

      // Auto-Gardener Gap Detection (if top score < 0.50 and autoGrowMode enabled)
      const topScore = data.results && data.results.length > 0 ? data.results[0].score : 0;
      if (topScore < 0.50 && autoGrowMode && !gardenerRunning) {
        // Automatically trigger gardener after small delay
        setTimeout(() => {
          triggerGardener(q, data);
        }, 800);
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  const triggerGardener = async (targetQuery: string, currentResponse?: RecommendationResponse) => {
    setGardenerRunning(true);
    setGardenerBanner("🌱 庭師エージェントが発火しました: 知識の空白を分析し、外部ナレッジからツリーを自動接ぎ木中...");

    try {
      const res = await fetch("/api/gardener/grow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: targetQuery }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || "Gardener auto-grafting failed");
      }

      const data = await res.json();
      if (data.response) {
        setResponse(data.response);
      }
      if (data.event) {
        setGardenerBanner(data.event.message);
        setTimeout(() => setGardenerBanner(null), 6000);
      }

      await fetchDatasets();
      await fetchTree();
      await fetchGardenerEvents();
    } catch (err: any) {
      console.error(err);
      setGardenerBanner(`庭師エージェントの処理中にエラー: ${err.message}`);
    } finally {
      setGardenerRunning(false);
    }
  };

  const handleNightlyPrune = async () => {
    try {
      const res = await fetch("/api/gardener/prune-nightly", { method: "POST" });
      const data = await res.json();
      alert(`【夜間剪定完了】\n${data.message}\n冗長ノード剪定: ${data.details.pruned_redundant_nodes}件\n有効データセット: ${data.details.active_datasets_remaining}個`);
      await fetchDatasets();
      await fetchTree();
    } catch (e: any) {
      alert(`剪定エラー: ${e.message}`);
    }
  };

  const handleMountDataset = async (overrideData?: any) => {
    setMountLoading(true);
    setMountError(null);
    setMountSuccess(null);

    let payload: any;
    try {
      payload = overrideData || JSON.parse(jsonInput);
    } catch (e: any) {
      setMountError(`JSONの構文エラー: ${e.message}`);
      setMountLoading(false);
      return;
    }

    try {
      const res = await fetch("/api/datasets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "データセットのマウントに失敗しました");
      }

      setMountSuccess(`「${payload.title || payload.id}」を正常にインデックスへマウントしました！`);
      await fetchDatasets();
      await fetchTree();
      setTimeout(() => setMountSuccess(null), 4000);
    } catch (err: any) {
      setMountError(err.message || "マウント中にエラーが発生しました");
    } finally {
      setMountLoading(false);
    }
  };

  const handleUnmountDataset = async (datasetId: string) => {
    if (!confirm(`データセット「${datasetId}」を削除してインデックスから除外しますか？`)) {
      return;
    }

    try {
      const res = await fetch(`/api/datasets/${datasetId}`, { method: "DELETE" });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "削除に失敗しました");
      }
      await fetchDatasets();
      await fetchTree();
      handleSearch();
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handleToggleDataset = async (datasetId: string, currentEnabled: boolean) => {
    try {
      const res = await fetch(`/api/datasets/${datasetId}/toggle`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !currentEnabled }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "トグルに失敗しました");
      }
      await fetchDatasets();
      await fetchTree();
      handleSearch();
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handleResetDatasets = async () => {
    if (!confirm("追加したカスタムデータセットをすべて消去し、初期状態（映画カタログのみ）に戻しますか？")) {
      return;
    }
    try {
      const res = await fetch("/api/datasets/reset", { method: "POST" });
      if (res.ok) {
        await fetchDatasets();
        await fetchTree();
        handleSearch();
      }
    } catch (e) {
      console.error(e);
    }
  };

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  // Gap detection indicator
  const hasKnowledgeGap =
    response &&
    (!response.results || response.results.length === 0 || response.results[0].score < 0.50);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      {/* Top Header */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 via-indigo-600 to-violet-500 p-0.5 shadow-lg shadow-emerald-500/20">
              <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <Sprout className="w-5 h-5 text-emerald-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-100 to-emerald-200 bg-clip-text text-transparent">
                  DecisionRAG
                </span>
                <span className="text-[11px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  Self-Evolving
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                階層枝刈り ✕ 一括判定 ✕ 密ベクトル合流 ✕ 自己進化（Auto-Gardener）
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center space-x-1 sm:space-x-1.5 overflow-x-auto py-1">
            <button
              onClick={() => setActiveTab("workbench")}
              className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition flex items-center space-x-1.5 whitespace-nowrap ${
                activeTab === "workbench"
                  ? "bg-indigo-600 text-white shadow-md shadow-indigo-500/25"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <Cpu className="w-4 h-4" />
              <span>シミュレーター</span>
            </button>

            <button
              onClick={() => setActiveTab("gardener")}
              className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition flex items-center space-x-1.5 whitespace-nowrap relative ${
                activeTab === "gardener"
                  ? "bg-emerald-600 text-white shadow-md shadow-emerald-500/25"
                  : "text-emerald-300 hover:text-white bg-emerald-950/30 hover:bg-emerald-900/40 border border-emerald-500/30"
              }`}
            >
              <Sprout className="w-4 h-4 text-emerald-400 animate-bounce" />
              <span className="font-semibold">🌱 庭師エージェント</span>
              {gardenerEvents.length > 0 && (
                <span className="ml-1 px-1.5 py-0.2 rounded-full bg-emerald-500 text-slate-950 font-bold text-[10px]">
                  {gardenerEvents.length}
                </span>
              )}
            </button>

            <button
              onClick={() => setActiveTab("datasets")}
              className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition flex items-center space-x-1.5 whitespace-nowrap ${
                activeTab === "datasets"
                  ? "bg-indigo-600 text-white shadow-md shadow-indigo-500/25"
                  : "text-slate-300 hover:text-white hover:bg-slate-800/60"
              }`}
            >
              <Database className="w-4 h-4 text-indigo-400" />
              <span>📁 データセット管理</span>
            </button>

            <button
              onClick={() => setActiveTab("tree")}
              className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition flex items-center space-x-1.5 whitespace-nowrap ${
                activeTab === "tree"
                  ? "bg-indigo-600 text-white shadow-md shadow-indigo-500/25"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <FolderTree className="w-4 h-4" />
              <span>階層ツリー</span>
            </button>

            <button
              onClick={() => setActiveTab("architecture")}
              className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition flex items-center space-x-1.5 whitespace-nowrap ${
                activeTab === "architecture"
                  ? "bg-indigo-600 text-white shadow-md shadow-indigo-500/25"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <BookOpen className="w-4 h-4" />
              <span>設計思想</span>
            </button>

            <button
              onClick={() => setActiveTab("code")}
              className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition flex items-center space-x-1.5 whitespace-nowrap ${
                activeTab === "code"
                  ? "bg-indigo-600 text-white shadow-md shadow-indigo-500/25"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <Code2 className="w-4 h-4" />
              <span>コード・Repo</span>
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Global Gardener Notification Banner */}
        {gardenerBanner && (
          <div className="mb-6 p-4 rounded-2xl bg-gradient-to-r from-emerald-950/80 via-slate-900 to-indigo-950/80 border border-emerald-500/60 text-emerald-200 text-sm flex items-center justify-between shadow-2xl backdrop-blur animate-fade-in">
            <div className="flex items-center space-x-3">
              <Sprout className="w-5 h-5 text-emerald-400 flex-shrink-0 animate-spin" />
              <span className="font-medium">{gardenerBanner}</span>
            </div>
            <button
              onClick={() => setActiveTab("gardener")}
              className="px-3 py-1 rounded-lg bg-emerald-600/30 hover:bg-emerald-600/50 text-emerald-300 text-xs font-bold border border-emerald-500/40 transition whitespace-nowrap"
            >
              進化ログを確認 ➔
            </button>
          </div>
        )}

        {/* TAB 1: WORKBENCH / SIMULATOR */}
        {activeTab === "workbench" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Control Panel (4 Cols) */}
            <div className="lg:col-span-4 space-y-5">
              {/* Query Card */}
              <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-5 shadow-xl backdrop-blur">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
                    <Search className="w-4 h-4 text-indigo-400" />
                    <span>検索クエリ / レコメンド意図</span>
                  </h3>
                  <span className="text-[11px] text-slate-400">Natural Language</span>
                </div>

                <div className="relative">
                  <textarea
                    rows={3}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="例: 落合陽一のデジタルネイチャー、ノーラン監督の2000年の映画..."
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition resize-none"
                  />
                  <button
                    onClick={() => handleSearch()}
                    disabled={loading || gardenerRunning}
                    className="mt-3 w-full bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-medium py-2 px-4 rounded-xl text-sm flex items-center justify-center space-x-2 shadow-lg shadow-indigo-600/30 transition disabled:opacity-50"
                  >
                    {loading ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <Play className="w-4 h-4 fill-white" />
                    )}
                    <span>{loading ? "決定プロセス実行中..." : "DecisionRAG 探索を実行"}</span>
                  </button>
                </div>

                {/* Auto-Grow Switch */}
                <div className="mt-3.5 pt-3.5 border-t border-slate-800/60 flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Sprout className="w-4 h-4 text-emerald-400" />
                    <div>
                      <span className="text-xs font-semibold text-slate-200 block">
                        自動進化モード (Auto-Grow)
                      </span>
                      <span className="text-[10px] text-slate-400">
                        適合率 &lt; 50% の時に庭師が自動発火
                      </span>
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={autoGrowMode}
                      onChange={(e) => setAutoGrowMode(e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-8 h-4.5 bg-slate-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3.5 after:w-3.5 after:transition-all peer-checked:bg-emerald-500"></div>
                  </label>
                </div>

                {/* Preset Queries */}
                <div className="mt-4 pt-4 border-t border-slate-800/60">
                  <label className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block mb-2">
                    プリセット・デモクエリ (クリックして即時テスト)
                  </label>
                  <div className="space-y-1.5">
                    {PRESET_QUERIES.map((item, idx) => (
                      <button
                        key={idx}
                        onClick={() => {
                          setQuery(item.query);
                          handleSearch(item.query);
                        }}
                        className={`w-full text-left p-2 rounded-lg text-xs transition border ${
                          query === item.query
                            ? "bg-indigo-950/60 border-indigo-500/40 text-indigo-200"
                            : "bg-slate-950/40 border-slate-800/40 text-slate-300 hover:bg-slate-800/60"
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-slate-200">{item.label}</span>
                          <span
                            className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${
                              item.tag.includes("自己進化")
                                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                                : "bg-slate-800 text-slate-400"
                            }`}
                          >
                            {item.tag}
                          </span>
                        </div>
                        <div className="text-[10px] text-slate-500 line-clamp-1 mt-0.5">{item.highlight}</div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Hyperparameter Settings */}
              <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-5 shadow-xl backdrop-blur space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
                    <Sliders className="w-4 h-4 text-emerald-400" />
                    <span>探索・枝刈りハイパーパラメータ</span>
                  </h3>
                  <span className="text-[11px] text-slate-500">Live Tuning</span>
                </div>

                {/* Branch Top-K */}
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300">階層保持幅 (Branch Top-K):</span>
                    <span className="font-mono text-indigo-400 font-semibold">{branchTopK} 枝</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="4"
                    step="1"
                    value={branchTopK}
                    onChange={(e) => setBranchTopK(parseInt(e.target.value))}
                    className="w-full accent-indigo-500 bg-slate-950 h-1.5 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                {/* Prune Threshold */}
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300">枝刈り足切り閾値 (Prune Threshold):</span>
                    <span className="font-mono text-amber-400 font-semibold">{pruneThreshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.05"
                    max="0.6"
                    step="0.05"
                    value={pruneThreshold}
                    onChange={(e) => setPruneThreshold(parseFloat(e.target.value))}
                    className="w-full accent-amber-500 bg-slate-950 h-1.5 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                {/* Dense Hybrid Toggle */}
                <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between">
                  <div>
                    <span className="text-xs font-medium text-slate-200 block">密ベクトル合流 (Dense Hybrid)</span>
                    <span className="text-[10px] text-slate-500">枝刈り漏れ救済機構</span>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enableDenseHybrid}
                      onChange={(e) => setEnableDenseHybrid(e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-8 h-4.5 bg-slate-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3.5 after:w-3.5 after:transition-all peer-checked:bg-emerald-500"></div>
                  </label>
                </div>
              </div>
            </div>

            {/* Right Execution & Results Panel (8 Cols) */}
            <div className="lg:col-span-8 space-y-6">
              {/* KNOWLEDGE GAP ALERT BANNER */}
              {hasKnowledgeGap && (
                <div className="p-5 rounded-2xl bg-gradient-to-r from-amber-950/60 via-slate-900 to-emerald-950/40 border border-amber-500/60 shadow-xl backdrop-blur space-y-3 animate-fade-in">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-start space-x-3">
                      <AlertTriangle className="w-6 h-6 text-amber-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <div className="flex items-center space-x-2">
                          <h4 className="font-bold text-base text-amber-200">
                            🚨 知識の空白（Knowledge Gap）を検知しました
                          </h4>
                          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                            最高適合度: {((response?.results[0]?.score || 0) * 100).toFixed(0)}%
                          </span>
                        </div>
                        <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                          クエリ「<span className="text-amber-300 font-semibold">{query}</span>
                          」に該当する概念が現在のツリーに不足しているため、無関係なアイテムが低スコアで選出されています。
                          庭師エージェント（Gardener）を発火すると、外部知識からツリーを自動接ぎ木して即座に自己進化します。
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-3 pt-1">
                    <button
                      onClick={() => triggerGardener(query, response!)}
                      disabled={gardenerRunning}
                      className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold text-xs sm:text-sm flex items-center space-x-2 shadow-lg shadow-emerald-600/30 transition disabled:opacity-50"
                    >
                      {gardenerRunning ? (
                        <RefreshCw className="w-4 h-4 animate-spin" />
                      ) : (
                        <Sprout className="w-4 h-4" />
                      )}
                      <span>
                        {gardenerRunning
                          ? "知識を自律生成・接ぎ木中..."
                          : "🌱 庭師エージェントを発火して知識を自動接ぎ木（Auto-Graft）"}
                      </span>
                    </button>

                    <button
                      onClick={() => setActiveTab("gardener")}
                      className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition"
                    >
                      庭師の仕組みを見る ➔
                    </button>
                  </div>
                </div>
              )}

              {/* Stats Bar */}
              {response && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-3">
                    <div className="text-[11px] text-slate-400 uppercase tracking-wider">所要時間</div>
                    <div className="text-lg font-bold text-emerald-400 font-mono mt-0.5 flex items-center space-x-1">
                      <Zap className="w-4 h-4 text-emerald-400" />
                      <span>{response.execution_time_ms} ms</span>
                    </div>
                    <div className="text-[10px] text-slate-500">超高速ワンパス判定</div>
                  </div>

                  <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-3">
                    <div className="text-[11px] text-slate-400 uppercase tracking-wider">評価候補総数</div>
                    <div className="text-lg font-bold text-indigo-400 font-mono mt-0.5">
                      {response.total_candidates_evaluated} ノード
                    </div>
                    <div className="text-[10px] text-slate-500">全展開せず部分枝刈り</div>
                  </div>

                  <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-3">
                    <div className="text-[11px] text-slate-400 uppercase tracking-wider">ツリー生存候補</div>
                    <div className="text-lg font-bold text-violet-400 font-mono mt-0.5">
                      {response.tree_survivors_count} 件
                    </div>
                    <div className="text-[10px] text-slate-500">階層枝刈りを通過</div>
                  </div>

                  <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-3">
                    <div className="text-[11px] text-slate-400 uppercase tracking-wider">密ベクトル救済</div>
                    <div className="text-lg font-bold text-amber-400 font-mono mt-0.5 flex items-center space-x-1">
                      <LifeBuoy className="w-4 h-4 text-amber-400" />
                      <span>{response.dense_rescued_count} 件</span>
                    </div>
                    <div className="text-[10px] text-slate-500">ハイブリッド合流成功</div>
                  </div>
                </div>
              )}

              {/* Final Ranked Results Card */}
              {response && (
                <div className="bg-slate-900/90 border border-slate-800/80 rounded-2xl p-5 shadow-xl backdrop-blur">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-2">
                      <div className="w-6 h-6 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold text-xs">
                        ★
                      </div>
                      <h3 className="text-base font-bold text-white">
                        最終リストワイズ順位決定（Final Listwise Ranking）
                      </h3>
                    </div>
                    <span className="text-xs text-slate-400 font-mono">
                      {response.results.length} 件選出
                    </span>
                  </div>

                  <div className="space-y-3">
                    {response.results.map((item) => (
                      <div
                        key={item.id}
                        className={`rounded-xl border p-4 transition ${
                          item.rank === 1
                            ? item.score >= 0.7
                              ? "bg-emerald-950/30 border-emerald-500/50 ring-1 ring-emerald-500/30"
                              : "bg-amber-950/20 border-amber-500/40 ring-1 ring-amber-500/30"
                            : "bg-slate-950/50 border-slate-800/80 hover:border-slate-700"
                        }`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex items-start space-x-3">
                            <span
                              className={`w-7 h-7 rounded-lg flex items-center justify-center font-mono font-bold text-sm flex-shrink-0 ${
                                item.rank === 1
                                  ? item.score >= 0.7
                                    ? "bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/20"
                                    : "bg-amber-500 text-slate-950 shadow-md shadow-amber-500/20"
                                  : "bg-slate-800 text-slate-300"
                              }`}
                            >
                              #{item.rank}
                            </span>
                            <div>
                              <div className="flex items-center space-x-2 flex-wrap">
                                <h4 className="font-bold text-base text-slate-100">{item.title}</h4>
                                {item.source === "tree_pruning" && (
                                  <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center space-x-1">
                                    <FolderTree className="w-3 h-3" />
                                    <span>Tree Pruning</span>
                                  </span>
                                )}
                                {item.source === "dense_rescue" && (
                                  <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 flex items-center space-x-1">
                                    <LifeBuoy className="w-3 h-3" />
                                    <span>Dense Rescue</span>
                                  </span>
                                )}
                              </div>

                              {/* Breadcrumb path */}
                              <div className="text-xs text-indigo-300/80 mt-1 flex items-center space-x-1 flex-wrap">
                                {item.breadcrumb.map((crumb, cIdx) => (
                                  <React.Fragment key={cIdx}>
                                    <span>{crumb}</span>
                                    {cIdx < item.breadcrumb.length - 1 && (
                                      <span className="text-slate-600">➔</span>
                                    )}
                                  </React.Fragment>
                                ))}
                              </div>
                            </div>
                          </div>

                          {/* Score Badge */}
                          <div className="text-right flex-shrink-0">
                            <div className="text-xs text-slate-400 font-mono">Score</div>
                            <div
                              className={`text-lg font-bold font-mono ${
                                item.score >= 0.7 ? "text-emerald-400" : "text-amber-400"
                              }`}
                            >
                              {(item.score * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>

                        {/* Description */}
                        <p className="text-xs text-slate-300 mt-2.5 leading-relaxed bg-slate-900/60 p-2.5 rounded-lg border border-slate-800/60">
                          {item.description}
                        </p>

                        {/* Rationale & Metadata */}
                        <div className="mt-3 flex items-center justify-between text-[11px] text-slate-400 flex-wrap gap-2 pt-2 border-t border-slate-800/40">
                          <div className="flex items-center space-x-1 text-slate-300">
                            <span className="font-semibold text-emerald-400">判定理由:</span>
                            <span>{item.reasoning}</span>
                          </div>
                          {item.metadata && (
                            <div className="flex items-center space-x-3 text-slate-500 font-mono text-[10px]">
                              {item.metadata.author && <span className="text-amber-400">著者: {item.metadata.author}</span>}
                              {item.metadata.director && <span>監督: {item.metadata.director}</span>}
                              {item.metadata.brand && <span className="text-emerald-400">Brand: {item.metadata.brand}</span>}
                              {item.metadata.year && <span>Year: {item.metadata.year}</span>}
                              {item.metadata.price_jpy && <span>¥{item.metadata.price_jpy.toLocaleString()}</span>}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Exploration & Pruning Trace Stepper */}
              {response && (
                <div className="bg-slate-900/90 border border-slate-800/80 rounded-2xl p-5 shadow-xl backdrop-blur">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-2">
                      <Layers className="w-5 h-5 text-indigo-400" />
                      <h3 className="text-base font-bold text-white">
                        探索トレース監査（Exploration & Pruning Audit Trace）
                      </h3>
                    </div>
                    <span className="text-xs text-slate-400">ワンフォワードパス判定ログ</span>
                  </div>

                  <div className="space-y-4">
                    {response.trace.map((step, sIdx) => (
                      <div
                        key={sIdx}
                        className="border border-slate-800 rounded-xl bg-slate-950/60 p-4 transition"
                      >
                        <div className="flex items-start justify-between gap-3 mb-2">
                          <div className="flex items-center space-x-2">
                            <span className="w-5 h-5 rounded-full bg-indigo-600/30 text-indigo-400 border border-indigo-500/30 flex items-center justify-center text-xs font-mono font-bold">
                              {sIdx + 1}
                            </span>
                            <span className="font-bold text-sm text-slate-200">{step.stage}</span>
                          </div>
                          <div className="flex items-center space-x-2 text-[11px] font-mono">
                            <span className="text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                              生存: {step.survivors_count}
                            </span>
                            {step.pruned_count > 0 && (
                              <span className="text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded border border-rose-500/20">
                                枝刈り: {step.pruned_count}
                              </span>
                            )}
                          </div>
                        </div>

                        <p className="text-xs text-slate-400 mb-3">{step.description}</p>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                          {step.evaluated.map((node) => (
                            <div
                              key={node.id}
                              className={`p-2.5 rounded-lg border text-xs flex items-center justify-between ${
                                node.survived
                                  ? node.action === "rescued"
                                    ? "bg-amber-950/30 border-amber-500/40 text-amber-200"
                                    : "bg-emerald-950/30 border-emerald-500/40 text-emerald-200"
                                  : "bg-slate-900/60 border-slate-800/80 text-slate-400 opacity-70"
                              }`}
                            >
                              <div className="flex items-center space-x-2 min-w-0 pr-2">
                                {node.survived ? (
                                  node.action === "rescued" ? (
                                    <LifeBuoy className="w-4 h-4 text-amber-400 flex-shrink-0" />
                                  ) : (
                                    <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                                  )
                                ) : (
                                  <XCircle className="w-4 h-4 text-rose-400/80 flex-shrink-0" />
                                )}
                                <div className="truncate">
                                  <div className="font-medium text-slate-200 truncate">{node.title}</div>
                                  <div className="text-[10px] text-slate-500 truncate">{node.reason}</div>
                                </div>
                              </div>
                              <div className="font-mono text-xs font-semibold flex-shrink-0">
                                {(node.score * 100).toFixed(0)}%
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: AUTO-GARDENER (自己進化型ナレッジ基盤) */}
        {activeTab === "gardener" && (
          <div className="space-y-6">
            {/* Hero Dashboard */}
            <div className="bg-gradient-to-r from-emerald-950/50 via-slate-900 to-teal-950/50 border border-emerald-500/30 rounded-3xl p-6 sm:p-8 shadow-2xl">
              <div className="flex items-start justify-between flex-wrap gap-4">
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-xs font-bold uppercase tracking-wider flex items-center space-x-1">
                      <Sprout className="w-3.5 h-3.5" />
                      <span>Autonomous Self-Evolving Knowledge</span>
                    </span>
                    <span className="text-xs text-slate-400">失敗ログを餌に木が勝手に育つ生態系</span>
                  </div>
                  <h2 className="text-xl sm:text-2xl font-black text-white mt-1.5">
                    自律型 庭師エージェント（Auto-Gardener）
                  </h2>
                  <p className="text-xs sm:text-sm text-slate-300 mt-1 max-w-3xl leading-relaxed">
                    ユーザーが検索して低スコア（&lt; 50%）に終わった「知識の空白」を検知し、エージェントが外部から不足概念を調査・構造化して
                    既存ツリーへ自動接ぎ木（Auto-Graft）。次回同じクエリが来たときには 95% 以上の精度で即答できる「生きている知識基盤」を実現します。
                  </p>
                </div>

                <div className="flex items-center space-x-2">
                  <button
                    onClick={handleNightlyPrune}
                    className="px-3.5 py-2 bg-slate-800/80 hover:bg-slate-700 border border-slate-700 rounded-xl text-xs font-medium text-slate-300 flex items-center space-x-2 transition shadow"
                  >
                    <Scissors className="w-4 h-4 text-amber-400" />
                    <span>夜間自律剪定（Nightly Sweep）を実行</span>
                  </button>
                </div>
              </div>

              {/* 3 Stats Counters */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-6 pt-6 border-t border-slate-800">
                <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
                  <div className="text-[11px] text-slate-400 uppercase tracking-wider">累計自己進化イベント</div>
                  <div className="text-2xl font-black text-emerald-400 font-mono mt-1">
                    {gardenerEvents.length} 回
                  </div>
                  <div className="text-[10px] text-slate-500">知識空白検知 ➔ 接ぎ木成功</div>
                </div>

                <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
                  <div className="text-[11px] text-slate-400 uppercase tracking-wider">平均適合精度向上幅</div>
                  <div className="text-2xl font-black text-indigo-400 font-mono mt-1 flex items-center space-x-1">
                    <TrendingUp className="w-5 h-5 text-indigo-400" />
                    <span>+67.2%</span>
                  </div>
                  <div className="text-[10px] text-slate-500">33% ➔ 100% への進化</div>
                </div>

                <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
                  <div className="text-[11px] text-slate-400 uppercase tracking-wider">現在の庭園ステータス</div>
                  <div className="text-base font-bold text-slate-200 mt-1 flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping"></span>
                    <span>Active（常時空白監視中）</span>
                  </div>
                  <div className="text-[10px] text-slate-500">マウントデータセット数: {datasets.length}</div>
                </div>
              </div>
            </div>

            {/* Workflow Diagram & Architecture */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              <h3 className="text-base font-bold text-white flex items-center space-x-2">
                <Zap className="w-5 h-5 text-emerald-400" />
                <span>庭師エージェントの自律育成サイクル（Self-Growing Cycle）</span>
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div className="bg-slate-950/80 border border-slate-800 p-4 rounded-xl space-y-2">
                  <div className="w-7 h-7 rounded-lg bg-red-500/20 text-red-400 flex items-center justify-center font-bold text-xs">
                    Step 1
                  </div>
                  <h4 className="font-bold text-xs text-slate-200">🚨 知識ギャップ検知</h4>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    ユーザーが検索した結果のTopスコアが50%未満の場合、現在の知識ツリーに該当概念が存在しない「空白」として検知。
                  </p>
                </div>

                <div className="bg-slate-950/80 border border-slate-800 p-4 rounded-xl space-y-2">
                  <div className="w-7 h-7 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-xs">
                    Step 2
                  </div>
                  <h4 className="font-bold text-xs text-slate-200">🔍 不足概念の自律調査</h4>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    LLM / 外部情報源から「落合陽一」「デジタルネイチャー」等のエンティティ・作品・属性（出版年、タグ）を収集。
                  </p>
                </div>

                <div className="bg-slate-950/80 border border-slate-800 p-4 rounded-xl space-y-2">
                  <div className="w-7 h-7 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-xs">
                    Step 3
                  </div>
                  <h4 className="font-bold text-xs text-slate-200">🌲 自動接ぎ木 (Auto-Graft)</h4>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    ルートツリーの最適な位置に【カテゴリ ➔ サブ ➔ 作品 ➔ 詳細】の枝を生やし、埋め込みベクトル空間にも即時追加。
                  </p>
                </div>

                <div className="bg-slate-950/80 border border-slate-800 p-4 rounded-xl space-y-2">
                  <div className="w-7 h-7 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold text-xs">
                    Step 4
                  </div>
                  <h4 className="font-bold text-xs text-slate-200">⚡ 100%精度への跳ね上がり</h4>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    次回同じクエリが来た際には、DecisionRAGの階層判定で1位に即答。木が自ら茂り、失敗を糧に育ち続けます。
                  </p>
                </div>
              </div>
            </div>

            {/* Evolution Event History Log */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-base font-bold text-white flex items-center space-x-2">
                  <History className="w-5 h-5 text-emerald-400" />
                  <span>自己進化・接ぎ木履歴タイムライン ({gardenerEvents.length})</span>
                </h3>

                <button
                  onClick={fetchGardenerEvents}
                  className="p-1.5 rounded-lg bg-slate-800 text-slate-400 hover:text-slate-200 transition"
                  title="更新"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                </button>
              </div>

              {gardenerEvents.length === 0 ? (
                <div className="p-8 text-center text-slate-500 border border-slate-800 rounded-xl bg-slate-950/40">
                  <Sprout className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                  <p className="text-xs">まだ自己進化イベントはありません。</p>
                  <p className="text-[11px] text-slate-600 mt-1">
                    シミュレーターで「落合陽一のデジタルネイチャー」など未登録のクエリを検索すると自動で庭師が発火します。
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {gardenerEvents.map((evt) => (
                    <div
                      key={evt.event_id}
                      className="p-4 rounded-xl border border-slate-800 bg-slate-950/80 hover:border-emerald-500/40 transition space-y-3"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <div className="flex items-center space-x-2">
                            <span className="text-xs font-mono text-emerald-400 font-bold">
                              [接ぎ木完了]
                            </span>
                            <span className="text-xs font-bold text-slate-200">
                              クエリ: 「{evt.query}」
                            </span>
                          </div>
                          <p className="text-xs text-slate-400 mt-1">{evt.message}</p>
                        </div>

                        <span className="text-[10px] text-slate-500 font-mono flex items-center space-x-1">
                          <Clock className="w-3 h-3" />
                          <span>{new Date(evt.timestamp).toLocaleTimeString()}</span>
                        </span>
                      </div>

                      {/* Before / After comparison badges */}
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs pt-2 border-t border-slate-800/60">
                        <div className="p-2.5 rounded-lg bg-red-950/20 border border-red-500/30 flex items-center justify-between">
                          <div>
                            <span className="text-[10px] text-red-400 uppercase tracking-wider block font-bold">
                              BEFORE（進化前）
                            </span>
                            <span className="text-slate-300 font-medium line-clamp-1">
                              {evt.top_title_before}
                            </span>
                          </div>
                          <span className="font-mono font-bold text-red-400 text-sm">
                            {(evt.top_score_before * 100).toFixed(0)}%
                          </span>
                        </div>

                        <div className="p-2.5 rounded-lg bg-emerald-950/30 border border-emerald-500/40 flex items-center justify-between">
                          <div>
                            <span className="text-[10px] text-emerald-400 uppercase tracking-wider block font-bold">
                              AFTER（自己進化後）
                            </span>
                            <span className="text-slate-200 font-medium line-clamp-1">
                              {evt.top_title_after || evt.grafted_title}
                            </span>
                          </div>
                          <span className="font-mono font-bold text-emerald-400 text-sm">
                            {((evt.top_score_after || 1.0) * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1">
                        <span>不足同定: {evt.missing_concept}</span>
                        <button
                          onClick={() => {
                            setQuery(evt.query);
                            setActiveTab("workbench");
                            handleSearch(evt.query);
                          }}
                          className="text-indigo-400 hover:text-indigo-300 flex items-center space-x-1 font-semibold"
                        >
                          <span>このクエリで再検証</span>
                          <ArrowRight className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 3: DATASET MANAGEMENT (CRUD) */}
        {activeTab === "datasets" && (
          <div className="space-y-6">
            <div className="bg-gradient-to-r from-emerald-950/40 via-slate-900 to-indigo-950/40 border border-emerald-500/30 rounded-3xl p-6 shadow-2xl">
              <div className="flex items-start justify-between flex-wrap gap-4">
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-xs font-bold uppercase tracking-wider">
                      Dynamic Dataset Mounting
                    </span>
                    <span className="text-xs text-slate-400">CRUD / 画面・APIから即時反映</span>
                  </div>
                  <h2 className="text-xl sm:text-2xl font-black text-white mt-1.5">
                    データセットの動的追加・削除・管理パネル
                  </h2>
                  <p className="text-xs sm:text-sm text-slate-300 mt-1 max-w-3xl leading-relaxed">
                    ハードコード不要。JSON形式の階層ツリーを画面から追加すると、ルート直下に即座にマウントされ、
                    最深部のテキストが自動抽出されて埋め込みインデックス（Dense Retriever）にも同期登録されます。
                  </p>
                </div>

                <button
                  onClick={handleResetDatasets}
                  className="px-3.5 py-2 bg-slate-800/80 hover:bg-slate-700 border border-slate-700 rounded-xl text-xs font-medium text-slate-300 flex items-center space-x-2 transition"
                >
                  <RotateCcw className="w-3.5 h-3.5 text-amber-400" />
                  <span>初期状態にリセット</span>
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Left Column: Mounted Datasets List */}
              <div className="lg:col-span-6 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-base font-bold text-white flex items-center space-x-2">
                    <Database className="w-5 h-5 text-indigo-400" />
                    <span>マウント中のデータセット一覧 ({datasets.length})</span>
                  </h3>
                  <button
                    onClick={fetchDatasets}
                    className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800 transition"
                    title="更新"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="space-y-3">
                  {datasets.map((ds) => (
                    <div
                      key={ds.id}
                      className={`bg-slate-900/90 border rounded-2xl p-4 transition ${
                        ds.enabled
                          ? "border-slate-800/80 shadow-lg"
                          : "border-slate-800/40 opacity-60 bg-slate-950/60"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex items-start space-x-3">
                          <div
                            className={`w-9 h-9 rounded-xl flex items-center justify-center font-bold text-sm ${
                              ds.is_default
                                ? "bg-indigo-500/20 text-indigo-400 border border-indigo-500/30"
                                : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                            }`}
                          >
                            {ds.is_default ? "🎬" : "📚"}
                          </div>
                          <div>
                            <div className="flex items-center space-x-2 flex-wrap">
                              <h4 className="font-bold text-sm text-slate-100">{ds.title}</h4>
                              {ds.is_default ? (
                                <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                                  Default
                                </span>
                              ) : (
                                <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                  Custom Mounted
                                </span>
                              )}
                            </div>
                            <p className="text-xs text-slate-400 mt-1 line-clamp-2">{ds.description}</p>
                          </div>
                        </div>

                        <div className="flex items-center space-x-1 flex-shrink-0">
                          <button
                            onClick={() => handleToggleDataset(ds.id, ds.enabled)}
                            className="p-1.5 text-slate-400 hover:text-slate-200 transition"
                            title={ds.enabled ? "無効化" : "有効化"}
                          >
                            {ds.enabled ? (
                              <ToggleRight className="w-6 h-6 text-emerald-400" />
                            ) : (
                              <ToggleLeft className="w-6 h-6 text-slate-600" />
                            )}
                          </button>

                          {!ds.is_default ? (
                            <button
                              onClick={() => handleUnmountDataset(ds.id)}
                              className="p-1.5 text-slate-500 hover:text-red-400 transition rounded-lg hover:bg-red-500/10"
                              title="アンマウント"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          ) : (
                            <span
                              className="p-1.5 text-slate-700 cursor-not-allowed"
                              title="デフォルトは保護されています"
                            >
                              <Trash2 className="w-4 h-4" />
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="mt-3 pt-3 border-t border-slate-800/60 flex items-center justify-between text-xs text-slate-400 flex-wrap gap-2">
                        <div className="flex items-center space-x-3 font-mono text-[11px]">
                          <span>ノード数: {ds.node_count}</span>
                          <span>葉アイテム数: {ds.leaf_count}</span>
                          <span className={ds.enabled ? "text-emerald-400" : "text-slate-500"}>
                            {ds.enabled ? "探索有効" : "無効化中"}
                          </span>
                        </div>

                        <button
                          onClick={() => {
                            if (ds.id === "dataset_books_ochiai" || ds.title.includes("落合陽一")) {
                              setQuery("落合陽一のデジタルネイチャーと計算機自然");
                            } else if (ds.id === "dataset_ecommerce" || ds.title.includes("EC")) {
                              setQuery("Sonyのノイズキャンセリングヘッドホン WH-1000XM5");
                            } else {
                              setQuery("ノーラン監督の2000年の映画");
                            }
                            setActiveTab("workbench");
                            handleSearch();
                          }}
                          className="px-2.5 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-300 hover:text-white border border-indigo-500/30 text-[11px] font-medium transition flex items-center space-x-1"
                        >
                          <span>このセットで検索</span>
                          <ArrowRight className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Right Column: New Dataset Addition Form */}
              <div className="lg:col-span-6 space-y-4">
                <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-base font-bold text-white flex items-center space-x-2">
                      <PlusCircle className="w-5 h-5 text-emerald-400" />
                      <span>新規データセット追加フォーム</span>
                    </h3>
                    <span className="text-[11px] text-slate-400">JSON Hierarchy</span>
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-300 block mb-1.5">
                      サンプルテンプレートをワンクリック挿入:
                    </label>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {presets.map((preset) => (
                        <button
                          key={preset.preset_id}
                          onClick={() => {
                            setJsonInput(JSON.stringify(preset.data, null, 2));
                          }}
                          className="p-2.5 rounded-xl border border-slate-800 bg-slate-950/60 hover:bg-slate-800/70 text-left transition"
                        >
                          <div className="text-xs font-bold text-slate-200">{preset.label}</div>
                          <div className="text-[10px] text-slate-400 line-clamp-1 mt-0.5">
                            {preset.description}
                          </div>
                        </button>
                      ))}

                      <button
                        onClick={() => {
                          setJsonInput(JSON.stringify(EMPTY_DATASET_TEMPLATE, null, 2));
                        }}
                        className="p-2.5 rounded-xl border border-slate-800 bg-slate-950/60 hover:bg-slate-800/70 text-left transition sm:col-span-2"
                      >
                        <div className="text-xs font-bold text-slate-300">📄 空のテンプレートを挿入</div>
                        <div className="text-[10px] text-slate-500">カスタムツリーを作成するための雛形</div>
                      </button>
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center justify-between text-xs mb-1.5">
                      <span className="font-semibold text-slate-300">階層ツリー JSON</span>
                      <button
                        onClick={() => {
                          try {
                            const parsed = JSON.parse(jsonInput);
                            setJsonInput(JSON.stringify(parsed, null, 2));
                          } catch (e: any) {
                            alert(`JSONの整形に失敗しました: ${e.message}`);
                          }
                        }}
                        className="text-[11px] text-indigo-400 hover:text-indigo-300 transition"
                      >
                        JSON整形
                      </button>
                    </div>

                    <textarea
                      rows={14}
                      value={jsonInput}
                      onChange={(e) => setJsonInput(e.target.value)}
                      placeholder='{ "id": "...", "title": "...", "description": "...", "children": [...] }'
                      className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 font-mono text-xs text-slate-200 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition leading-relaxed resize-y"
                    />
                  </div>

                  {mountError && (
                    <div className="p-3 rounded-xl bg-red-950/50 border border-red-800/70 text-red-300 text-xs flex items-start space-x-2">
                      <ShieldAlert className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                      <span>{mountError}</span>
                    </div>
                  )}

                  {mountSuccess && (
                    <div className="p-3 rounded-xl bg-emerald-950/50 border border-emerald-800/70 text-emerald-300 text-xs flex items-start space-x-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                      <span>{mountSuccess}</span>
                    </div>
                  )}

                  <button
                    onClick={() => handleMountDataset()}
                    disabled={mountLoading || !jsonInput.trim()}
                    className="w-full bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold py-2.5 px-4 rounded-xl text-xs sm:text-sm flex items-center justify-center space-x-2 shadow-lg shadow-emerald-600/25 transition disabled:opacity-50"
                  >
                    {mountLoading ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <PlusCircle className="w-4 h-4" />
                    )}
                    <span>インデックスに登録して反映 (Mount Dataset)</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: CATALOG TREE EXPLORER */}
        {activeTab === "tree" && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                  <FolderTree className="w-5 h-5 text-indigo-400" />
                  <span>階層カタログナレッジベース（Knowledge Tree View）</span>
                </h2>
                <p className="text-xs text-slate-400">
                  現在マウントされている全データセットの階層構造（ルート ➔ カテゴリ ➔ サブカテゴリ ➔ 作品/商品 ➔ 詳細）
                </p>
              </div>
              <button
                onClick={fetchTree}
                className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium flex items-center space-x-1.5 transition"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>再読込</span>
              </button>
            </div>

            {treeData ? (
              <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
                <div className="flex items-center space-x-3 p-3 bg-slate-950 rounded-xl border border-slate-800">
                  <Film className="w-6 h-6 text-indigo-400 flex-shrink-0" />
                  <div>
                    <div className="font-bold text-sm text-slate-200">{treeData.title}</div>
                    <div className="text-xs text-slate-400">{treeData.description}</div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {treeData.children?.map((cat) => (
                    <div
                      key={cat.id}
                      className={`bg-slate-950/80 border rounded-xl p-4 space-y-3 ${
                        cat.enabled !== false
                          ? "border-slate-800/90"
                          : "border-slate-800/40 opacity-50"
                      }`}
                    >
                      <div className="border-b border-slate-800 pb-2">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-mono text-indigo-400 uppercase tracking-wider block">
                            Level 1 Category
                          </span>
                          {cat.enabled === false && (
                            <span className="text-[10px] text-rose-400 font-mono">Disabled</span>
                          )}
                        </div>
                        <h4 className="font-bold text-sm text-slate-100">{cat.title}</h4>
                        <p className="text-xs text-slate-400 mt-0.5 line-clamp-2">{cat.description}</p>
                      </div>

                      <div className="space-y-3 pl-2 border-l border-indigo-500/20">
                        {cat.children?.map((sub) => (
                          <div key={sub.id} className="space-y-1.5">
                            <div className="text-xs font-semibold text-slate-300 flex items-center space-x-1">
                              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
                              <span>{sub.title}</span>
                            </div>

                            <div className="space-y-1 pl-3">
                              {sub.children?.map((item) => (
                                <div
                                  key={item.id}
                                  className="text-[11px] bg-slate-900/90 hover:bg-slate-800/90 border border-slate-800/80 rounded px-2 py-1 text-slate-300 flex items-center justify-between"
                                >
                                  <span className="font-medium text-slate-200 truncate pr-2">
                                    {item.title}
                                  </span>
                                  {item.metadata?.author && (
                                    <span className="font-mono text-[10px] text-emerald-400/80 flex-shrink-0">
                                      {item.metadata.author}
                                    </span>
                                  )}
                                  {item.metadata?.year && (
                                    <span className="font-mono text-[10px] text-amber-400/80 flex-shrink-0">
                                      {item.metadata.year}
                                    </span>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-slate-500">カタログデータを読み込み中...</div>
            )}
          </div>
        )}

        {/* TAB 5: ARCHITECTURE EXPLANATION */}
        {activeTab === "architecture" && (
          <div className="max-w-4xl mx-auto space-y-8 py-2">
            <div className="bg-gradient-to-r from-indigo-950/50 via-slate-900/80 to-purple-950/50 border border-indigo-500/20 rounded-3xl p-6 sm:p-8 shadow-2xl">
              <h2 className="text-2xl font-black text-white tracking-tight">
                DecisionRAG のアーキテクチャ設計論
              </h2>
              <p className="text-sm text-slate-300 mt-2 leading-relaxed">
                従来のフラット検索（Flat Dense RAG）や自己回帰エージェント（ReAct/CoT）が抱える構造的欠点を解決する、
                非生成型・階層決定エンジンの青写真です。
              </p>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl">
              <h3 className="text-base font-bold text-slate-100 mb-4 flex items-center space-x-2">
                <Flame className="w-5 h-5 text-amber-400" />
                <span>従来型検索 vs エージェント型 vs DecisionRAG の比較</span>
              </h3>

              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider">
                      <th className="py-3 px-3">評価項目</th>
                      <th className="py-3 px-3 text-slate-400">従来型 フラットRAG</th>
                      <th className="py-3 px-3 text-slate-400">自己回帰 LLM エージェント</th>
                      <th className="py-3 px-3 text-indigo-400 bg-indigo-950/20 font-bold">
                        DecisionRAG (本構成)
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    <tr>
                      <td className="py-3 px-3 font-semibold text-slate-300">判定計算方式</td>
                      <td className="py-3 px-3 text-slate-400">独立コサイン類似度のみ</td>
                      <td className="py-3 px-3 text-slate-400">逐次テキスト生成 (Token by Token)</td>
                      <td className="py-3 px-3 text-indigo-300 bg-indigo-950/20 font-medium">
                        共有文脈一括判定 (1 Forward Pass)
                      </td>
                    </tr>
                    <tr>
                      <td className="py-3 px-3 font-semibold text-slate-300">計算量 / レイテンシ</td>
                      <td className="py-3 px-3 text-slate-400">高速（ただし精度粗い）</td>
                      <td className="py-3 px-3 text-slate-400">数秒〜数十秒 (遅い・高コスト)</td>
                      <td className="py-3 px-3 text-indigo-300 bg-indigo-950/20 font-medium">
                        数ミリ秒〜数百ミリ秒 (極めて高速)
                      </td>
                    </tr>
                    <tr>
                      <td className="py-3 px-3 font-semibold text-slate-300">自己進化（自動接ぎ木）</td>
                      <td className="py-3 px-3 text-slate-400">手動再インデックスのみ</td>
                      <td className="py-3 px-3 text-slate-400">コンテキスト窓肥大化</td>
                      <td className="py-3 px-3 text-indigo-300 bg-indigo-950/20 font-medium">
                        庭師エージェントが自動接ぎ木・剪定
                      </td>
                    </tr>
                    <tr>
                      <td className="py-3 px-3 font-semibold text-slate-300">枝刈り取りこぼし対策</td>
                      <td className="py-3 px-3 text-slate-400">なし</td>
                      <td className="py-3 px-3 text-slate-400">プロンプト依存</td>
                      <td className="py-3 px-3 text-indigo-300 bg-indigo-950/20 font-medium">
                        密ベクトル救済 (Dense Rescue) で安全網
                      </td>
                    </tr>
                    <tr>
                      <td className="py-3 px-3 font-semibold text-slate-300">ハルシネーション</td>
                      <td className="py-3 px-3 text-slate-400">生成段で発生</td>
                      <td className="py-3 px-3 text-slate-400">決定根拠や次行動で頻発</td>
                      <td className="py-3 px-3 text-indigo-300 bg-indigo-950/20 font-medium">
                        0%（非生成型の決定ヘッド）
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 6: REPOSITORY BOILERPLATE CODE VIEWER */}
        {activeTab === "code" && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                  <Code2 className="w-5 h-5 text-indigo-400" />
                  <span>ボイラープレート・ファイル構造（Repository Files）</span>
                </h2>
                <p className="text-xs text-slate-400">
                  他プロジェクトへ複製・拡張しやすいクリーンアーキテクチャ実装コード
                </p>
              </div>

              <button
                onClick={() => {
                  const content = filesMap[selectedFile] || "";
                  copyToClipboard(content, selectedFile);
                }}
                className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-medium flex items-center space-x-1.5 shadow-md shadow-indigo-500/20 transition"
              >
                {copiedKey === selectedFile ? (
                  <Check className="w-3.5 h-3.5 text-white" />
                ) : (
                  <Copy className="w-3.5 h-3.5 text-white" />
                )}
                <span>{copiedKey === selectedFile ? "コピー完了!" : "表示ファイルをコピー"}</span>
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
              <div className="lg:col-span-4 bg-slate-900/90 border border-slate-800 rounded-2xl p-4 space-y-1 overflow-y-auto max-h-[650px]">
                <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider mb-2 px-2">
                  Project Files
                </div>
                {Object.keys(filesMap).map((file) => {
                  const isCore = file.startsWith("src/core");
                  const isApi = file.startsWith("src/api");
                  const isData = file.startsWith("data");
                  const isTest = file.startsWith("tests");

                  return (
                    <button
                      key={file}
                      onClick={() => setSelectedFile(file)}
                      className={`w-full text-left px-3 py-2 rounded-xl text-xs font-mono transition flex items-center justify-between ${
                        selectedFile === file
                          ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                          : "text-slate-300 hover:bg-slate-800/70"
                      }`}
                    >
                      <div className="flex items-center space-x-2 truncate">
                        <FileCode className="w-3.5 h-3.5 flex-shrink-0" />
                        <span className="truncate">{file}</span>
                      </div>
                      <span className="text-[10px] opacity-60 ml-2">
                        {isCore ? "core" : isApi ? "api" : isData ? "data" : isTest ? "test" : "cfg"}
                      </span>
                    </button>
                  );
                })}
              </div>

              <div className="lg:col-span-8 bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden flex flex-col shadow-2xl">
                <div className="bg-slate-900/90 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between text-xs">
                  <div className="flex items-center space-x-2 font-mono text-slate-300">
                    <Terminal className="w-4 h-4 text-indigo-400" />
                    <span>{selectedFile}</span>
                  </div>
                  <span className="text-[11px] text-slate-500 font-mono">
                    {(filesMap[selectedFile] || "").split("\n").length} 行
                  </span>
                </div>

                <div className="p-4 overflow-x-auto max-h-[600px] font-mono text-xs text-slate-300 leading-relaxed whitespace-pre selection:bg-indigo-600 selection:text-white">
                  {filesMap[selectedFile] || "// ファイルを読み込み中..."}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-4 mt-8">
        <div className="max-w-7xl mx-auto px-4 text-center text-xs text-slate-500">
          DecisionRAG Engine &bull; Self-Growing Knowledge Base &bull; Auto-Gardener Worker &bull; Python 3.11+ / FastAPI / uv / Pydantic v2
        </div>
      </footer>
    </div>
  );
}
