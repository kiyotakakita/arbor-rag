import express from "express";
import { execFile } from "child_process";
import path from "path";
import fs from "fs";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";

const app = express();
const PORT = 3000;
const isProd = process.env.NODE_ENV === "production";

app.use(express.json({ limit: "10mb" }));

const DATASETS_DIR = path.resolve(process.cwd(), "data", "datasets");
if (!fs.existsSync(DATASETS_DIR)) {
  fs.mkdirSync(DATASETS_DIR, { recursive: true });
}

const EVENTS_FILE = path.resolve(process.cwd(), "data", "gardener_events.json");

function loadGardenerEvents(): any[] {
  if (fs.existsSync(EVENTS_FILE)) {
    try {
      return JSON.parse(fs.readFileSync(EVENTS_FILE, "utf-8"));
    } catch (e) {
      return [];
    }
  }
  return [];
}

function saveGardenerEvent(event: any) {
  const events = loadGardenerEvents();
  events.unshift(event);
  fs.writeFileSync(EVENTS_FILE, JSON.stringify(events.slice(0, 50), null, 2), "utf-8");
}

// API: Get Full Tree Structure (including mounted datasets)
app.get("/api/tree", (req, res) => {
  const scriptPath = path.resolve(process.cwd(), "src", "cli_json.py");
  execFile("python3", [scriptPath, "--tree"], { cwd: process.cwd() }, (error, stdout, stderr) => {
    if (error) {
      console.error("Python CLI tree error:", stderr);
      const catalogPath = path.resolve(process.cwd(), "data", "sample_catalog.json");
      if (fs.existsSync(catalogPath)) {
        return res.json(JSON.parse(fs.readFileSync(catalogPath, "utf-8")));
      }
      return res.status(500).json({ error: stderr || error.message });
    }
    try {
      const data = JSON.parse(stdout);
      res.json(data);
    } catch (err) {
      res.status(500).json({ error: "Failed to parse tree JSON" });
    }
  });
});

// API: List Mounted Datasets
app.get("/api/datasets", (req, res) => {
  const scriptPath = path.resolve(process.cwd(), "src", "cli_json.py");
  execFile("python3", [scriptPath, "--datasets"], { cwd: process.cwd() }, (error, stdout, stderr) => {
    if (error) {
      console.error("Python CLI datasets error:", stderr);
      return res.status(500).json({ error: stderr || error.message });
    }
    try {
      const datasets = JSON.parse(stdout);
      res.json({ datasets });
    } catch (err) {
      res.status(500).json({ error: "Failed to parse datasets list" });
    }
  });
});

// API: Mount New Dataset
app.post("/api/datasets", (req, res) => {
  const payload = req.body;
  if (!payload || typeof payload !== "object") {
    return res.status(400).json({ error: "Invalid JSON payload" });
  }

  const title =
    payload.title ||
    (payload.root_category && payload.root_category.title) ||
    "";
  if (!title.trim()) {
    return res.status(422).json({ error: "データセットには 'title' (タイトル) が必須です。" });
  }

  const rawId = payload.id || `custom_${Date.now()}`;
  const safeId = String(rawId).replace(/[^a-zA-Z0-9_-]/g, "_");
  payload.id = safeId;
  payload.enabled = payload.enabled !== false;

  const targetPath = path.join(DATASETS_DIR, `${safeId}.json`);
  fs.writeFileSync(targetPath, JSON.stringify(payload, null, 2), "utf-8");

  const scriptPath = path.resolve(process.cwd(), "src", "cli_json.py");
  execFile("python3", [scriptPath, "--datasets"], { cwd: process.cwd() }, (error, stdout, stderr) => {
    if (error) {
      console.error("Mount validation error:", stderr);
      if (fs.existsSync(targetPath)) fs.unlinkSync(targetPath);
      return res.status(400).json({ error: `データセットの解析に失敗しました: ${stderr || error.message}` });
    }
    try {
      const datasets = JSON.parse(stdout);
      res.json({ status: "success", dataset_id: safeId, datasets });
    } catch (e) {
      res.json({ status: "success", dataset_id: safeId });
    }
  });
});

// API: Unmount Dataset
app.delete("/api/datasets/:id", (req, res) => {
  const datasetId = req.params.id;
  if (datasetId === "default_movies") {
    return res.status(400).json({ error: "デフォルトの映画カタログは削除できません。トグルで無効化してください。" });
  }

  const safeId = String(datasetId).replace(/[^a-zA-Z0-9_-]/g, "_");
  const targetPath = path.join(DATASETS_DIR, `${safeId}.json`);

  if (fs.existsSync(targetPath)) {
    fs.unlinkSync(targetPath);
  }

  const scriptPath = path.resolve(process.cwd(), "src", "cli_json.py");
  execFile("python3", [scriptPath, "--datasets"], { cwd: process.cwd() }, (error, stdout, stderr) => {
    try {
      const datasets = JSON.parse(stdout);
      res.json({ status: "success", unmounted: datasetId, datasets });
    } catch (e) {
      res.json({ status: "success", unmounted: datasetId });
    }
  });
});

// API: Toggle Dataset
app.patch("/api/datasets/:id/toggle", (req, res) => {
  const datasetId = req.params.id;
  const { enabled = true } = req.body;
  const safeId = String(datasetId).replace(/[^a-zA-Z0-9_-]/g, "_");
  const targetPath = path.join(DATASETS_DIR, `${safeId}.json`);

  if (fs.existsSync(targetPath)) {
    try {
      const data = JSON.parse(fs.readFileSync(targetPath, "utf-8"));
      data.enabled = !!enabled;
      fs.writeFileSync(targetPath, JSON.stringify(data, null, 2), "utf-8");
    } catch (e) {
      return res.status(500).json({ error: "Failed to update dataset file" });
    }
  }

  const scriptPath = path.resolve(process.cwd(), "src", "cli_json.py");
  execFile("python3", [scriptPath, "--datasets"], { cwd: process.cwd() }, (error, stdout, stderr) => {
    try {
      const datasets = JSON.parse(stdout);
      res.json({ status: "success", dataset_id: datasetId, enabled, datasets });
    } catch (e) {
      res.json({ status: "success", dataset_id: datasetId, enabled });
    }
  });
});

// API: Reset Datasets to Pristine State
app.post("/api/datasets/reset", (req, res) => {
  if (fs.existsSync(DATASETS_DIR)) {
    const files = fs.readdirSync(DATASETS_DIR);
    for (const file of files) {
      if (file.endsWith(".json")) {
        fs.unlinkSync(path.join(DATASETS_DIR, file));
      }
    }
  }

  const scriptPath = path.resolve(process.cwd(), "src", "cli_json.py");
  execFile("python3", [scriptPath, "--datasets"], { cwd: process.cwd() }, (error, stdout, stderr) => {
    try {
      const datasets = JSON.parse(stdout);
      res.json({ status: "success", message: "Reset to default movies catalog", datasets });
    } catch (e) {
      res.json({ status: "success" });
    }
  });
});

// API: Dataset Presets
app.get("/api/datasets/presets", (req, res) => {
  const presets = [];
  const baseDir = path.resolve(process.cwd(), "data");

  const ochiaiPath = path.join(baseDir, "sample_books_ochiai.json");
  if (fs.existsSync(ochiaiPath)) {
    try {
      presets.push({
        preset_id: "ochiai_books",
        label: "📚 落合陽一 著作・デジタル思想コレクション",
        description: "『デジタルネイチャー』『魔法の世紀』『ビヨンド・デジタル・ネイチャー』等のメディア論書籍群",
        data: JSON.parse(fs.readFileSync(ochiaiPath, "utf-8")),
      });
    } catch (e) {}
  }

  const ecPath = path.join(baseDir, "sample_ecommerce.json");
  if (fs.existsSync(ecPath)) {
    try {
      presets.push({
        preset_id: "ecommerce_gadgets",
        label: "🛍️ EC商品・スマート家電＆ガジェット",
        description: "Sony WH-1000XM5等のオーディオ・高機能スマート製品カタログ",
        data: JSON.parse(fs.readFileSync(ecPath, "utf-8")),
      });
    } catch (e) {}
  }

  res.json({ presets });
});

// =============================================================================
// Auto-Gardener (自己進化型ナレッジ & 庭師エージェント)
// =============================================================================

app.get("/api/gardener/events", (req, res) => {
  res.json({ events: loadGardenerEvents() });
});

app.post("/api/gardener/grow", async (req, res) => {
  const { query, custom_branch } = req.body;
  if (!query) {
    return res.status(400).json({ error: "Missing query" });
  }

  let branchToGraft = custom_branch;

  // If valid Gemini API Key is available, synthesize branch using LLM with 4s timeout
  const apiKey = process.env.GEMINI_API_KEY;
  if (!branchToGraft && apiKey && !apiKey.startsWith("MY_") && apiKey.length > 10) {
    try {
      const ai = new GoogleGenAI();
      const prompt = `You are the Auto-Gardener agent for a hierarchical DecisionRAG knowledge engine.
The query "${query}" failed or has a knowledge gap in the current catalog.
Synthesize a high-quality hierarchical JSON tree branch to fill this gap.
Required format:
{
  "id": "grafted_${Date.now()}",
  "title": "Category title in Japanese (e.g. 書籍・学術：落合陽一著作コレクション)",
  "description": "Comprehensive category overview used by decision models for branch pruning",
  "children": [
    {
      "id": "sub_1",
      "title": "Subcategory title",
      "description": "Subcategory description",
      "metadata": { "domain": "..." },
      "children": [
        {
          "id": "item_1",
          "title": "Target Item / Work / Book / Product Name",
          "description": "Concrete summary matching user query",
          "metadata": {
            "author": "...",
            "year": 2024,
            "tags": ["..."],
            "keywords": ["..."]
          },
          "children": [
            {
              "id": "detail_1",
              "title": "...：詳細要約・核心論点",
              "description": "Deep passage explanation"
            }
          ]
        }
      ]
    }
  ]
}
Return JSON only without markdown fences.`;

      const fetchPromise = ai.models.generateContent({
        model: "gemini-3.8-flash",
        contents: prompt,
        config: {
          responseMimeType: "application/json",
        },
      });

      const timeoutPromise = new Promise<null>((_, reject) =>
        setTimeout(() => reject(new Error("Gemini API timeout (4s)")), 4000)
      );

      const geminiRes: any = await Promise.race([fetchPromise, timeoutPromise]);

      if (geminiRes && geminiRes.text) {
        branchToGraft = JSON.parse(geminiRes.text.trim());
      }
    } catch (llmErr) {
      console.warn("Gemini synthesis fallback to internal gardener:", llmErr);
    }
  }

  // Execute python gardener CLI
  const scriptPath = path.resolve(process.cwd(), "src", "cli_json.py");
  const args = [scriptPath, "--gardener-grow", String(query)];
  if (branchToGraft) {
    args.push("--gardener-branch-json", JSON.stringify(branchToGraft));
  }

  execFile("python3", args, { cwd: process.cwd() }, (error, stdout, stderr) => {
    if (error) {
      console.error("Gardener CLI error:", stderr);
      return res.status(500).json({ error: stderr || error.message });
    }
    try {
      const output = JSON.parse(stdout);
      if (output.event) {
        saveGardenerEvent(output.event);
      }
      res.json(output);
    } catch (parseErr) {
      console.error("Gardener parse error:", parseErr, stdout);
      res.status(500).json({ error: "Failed to parse Gardener output", raw: stdout });
    }
  });
});

app.post("/api/gardener/prune-nightly", (req, res) => {
  // Nightly sweep: deduplicate & optimize
  const files = fs.existsSync(DATASETS_DIR) ? fs.readdirSync(DATASETS_DIR) : [];
  let mergedCount = 0;
  let prunedCount = 0;

  // Clean empty or corrupt files
  for (const f of files) {
    if (f.endsWith(".json")) {
      const fp = path.join(DATASETS_DIR, f);
      try {
        const content = JSON.parse(fs.readFileSync(fp, "utf-8"));
        if (!content.children || content.children.length === 0) {
          fs.unlinkSync(fp);
          prunedCount++;
        }
      } catch (e) {
        fs.unlinkSync(fp);
        prunedCount++;
      }
    }
  }

  res.json({
    status: "success",
    timestamp: new Date().toISOString(),
    message: "夜間自律剪定（Nightly Gardener Sweep）が完了しました。",
    details: {
      pruned_redundant_nodes: prunedCount,
      merged_categories: mergedCount,
      active_datasets_remaining: fs.readdirSync(DATASETS_DIR).length + 1,
    },
  });
});

// API: DecisionRAG Recommend Execution
app.post("/api/recommend", (req, res) => {
  const {
    query = "ノーラン監督の2000年の映画",
    top_k = 5,
    branch_top_k = 2,
    prune_threshold = 0.25,
    enable_dense_hybrid = true,
    dense_top_n = 3,
  } = req.body;

  const scriptPath = path.resolve(process.cwd(), "src", "cli_json.py");
  const args = [
    scriptPath,
    "--query",
    String(query),
    "--top_k",
    String(top_k),
    "--branch_top_k",
    String(branch_top_k),
    "--prune_threshold",
    String(prune_threshold),
    "--enable_dense_hybrid",
    enable_dense_hybrid ? "true" : "false",
    "--dense_top_n",
    String(dense_top_n),
  ];

  execFile("python3", args, { cwd: process.cwd() }, (error, stdout, stderr) => {
    if (error) {
      console.error("Python CLI error:", stderr);
      return res.status(500).json({ error: stderr || error.message });
    }
    try {
      const output = JSON.parse(stdout);
      res.json(output);
    } catch (parseErr) {
      console.error("JSON parse error:", parseErr, stdout);
      res.status(500).json({ error: "Failed to parse Python engine output", raw: stdout });
    }
  });
});

// API: List and read repository boilerplate files for inspection & export
app.get("/api/files", (req, res) => {
  const allowedFiles = [
    "README.md",
    "pyproject.toml",
    ".env.example",
    "demo.py",
    "data/sample_catalog.json",
    "data/sample_books_ochiai.json",
    "src/config.py",
    "src/core/tree.py",
    "src/core/decision_model.py",
    "src/core/dense_retriever.py",
    "src/core/engine.py",
    "src/core/gardener.py",
    "src/schemas/recommendation.py",
    "src/api/app.py",
    "tests/test_tree.py",
    "tests/test_engine.py",
  ];

  const results: Record<string, string> = {};
  for (const f of allowedFiles) {
    const fullPath = path.resolve(process.cwd(), f);
    if (fs.existsSync(fullPath)) {
      results[f] = fs.readFileSync(fullPath, "utf-8");
    }
  }

  res.json({ files: results });
});

// API: Health Check
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", engine: "DecisionRAG", timestamp: new Date().toISOString() });
});

async function startServer() {
  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.resolve(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`DecisionRAG Engine server running at http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
