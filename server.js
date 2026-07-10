import express from "express";
import multer from "multer";
import path from "path";
import { fileURLToPath } from "url";
import Anthropic from "@anthropic-ai/sdk";

const here = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 30 * 1024 * 1024 }, // PDFのAPI上限(32MB)より少し小さく
});

const client = new Anthropic();

const MODEL = process.env.CLAUDE_MODEL || "claude-opus-4-8";

// 抽出結果のスキーマ(structured outputsで強制)
const EXTRACTION_SCHEMA = {
  type: "object",
  properties: {
    contract_title: {
      type: "string",
      description: "契約書のタイトル(例: 業務委託契約書、賃貸借契約書)",
    },
    counterparty: {
      type: "string",
      description:
        "取引先名(相手方の会社名・氏名)。自社と思われる当事者ではなく、支払先/請求元となる相手方を記載",
    },
    our_party: {
      type: ["string", "null"],
      description: "自社と思われる当事者名。判別できなければnull",
    },
    contract_date: {
      type: ["string", "null"],
      description: "契約締結日 (YYYY-MM-DD)。不明ならnull",
    },
    start_date: {
      type: ["string", "null"],
      description: "契約期間の開始日 (YYYY-MM-DD)。不明ならnull",
    },
    end_date: {
      type: ["string", "null"],
      description: "契約期間の終了日 (YYYY-MM-DD)。自動更新等で定めがなければnull",
    },
    amount_excl_tax: {
      type: ["number", "null"],
      description: "契約金額(税抜、円)。数値のみ。不明ならnull",
    },
    tax_amount: {
      type: ["number", "null"],
      description: "消費税額(円)。記載がなく計算もできなければnull",
    },
    amount_incl_tax: {
      type: ["number", "null"],
      description: "契約金額(税込、円)。税抜額と消費税から計算できる場合は計算する",
    },
    payment_frequency: {
      type: "string",
      enum: ["一括", "月額", "年額", "分割", "その他", "不明"],
      description: "支払形態",
    },
    payment_terms: {
      type: ["string", "null"],
      description: "支払条件(例: 月末締め翌月末払い、銀行振込)。不明ならnull",
    },
    account_title: {
      type: "string",
      description:
        "推奨される勘定科目。契約内容から最も適切なものを選ぶ(例: 外注費、業務委託費、支払手数料、地代家賃、賃借料、リース料、保険料、通信費、広告宣伝費、顧問料、修繕費、保守料、ソフトウェア、前払費用)",
    },
    account_title_reason: {
      type: "string",
      description: "その勘定科目を選んだ理由(1文で簡潔に)",
    },
    summary: {
      type: "string",
      description: "契約内容の要約(摘要欄に使える50字程度の簡潔な説明)",
    },
    confidence_notes: {
      type: ["string", "null"],
      description:
        "抽出に不確実な点があれば注意書き(例: 金額が複数記載、日付が不鮮明)。なければnull",
    },
  },
  required: [
    "contract_title",
    "counterparty",
    "our_party",
    "contract_date",
    "start_date",
    "end_date",
    "amount_excl_tax",
    "tax_amount",
    "amount_incl_tax",
    "payment_frequency",
    "payment_terms",
    "account_title",
    "account_title_reason",
    "summary",
    "confidence_notes",
  ],
  additionalProperties: false,
};

// カスタムフォーマット用: 項目名と値のペアの配列で返させる
const ITEMS_SCHEMA = {
  type: "object",
  properties: {
    items: {
      type: "array",
      description: "フォーマットに定義された項目を、定義された順に並べた配列",
      items: {
        type: "object",
        properties: {
          label: {
            type: "string",
            description: "フォーマットに書かれた項目名(原文のまま)",
          },
          value: {
            type: ["string", "null"],
            description: "契約書から読み取った値。書面から読み取れない場合はnull",
          },
        },
        required: ["label", "value"],
        additionalProperties: false,
      },
    },
  },
  required: ["items"],
  additionalProperties: false,
};

const SYSTEM_PROMPT = `あなたは日本の経理実務に詳しい契約書レビューアシスタントです。
アップロードされた契約書から経理処理に必要な情報を正確に抽出します。

ルール:
- 書面に実際に記載されている情報のみを抽出し、推測で値を作らない(勘定科目の推奨は例外)
- 金額は数値(円)で返す。カンマや「円」は含めない
- 複数の金額がある場合は契約の主たる対価を採用し、confidence_notesに補足を書く
- 日付は和暦でもYYYY-MM-DDに変換する
- 勘定科目は日本の一般的な中小企業の勘定科目体系から契約の実態に即して選ぶ`;

function fileToContentBlock(file) {
  const mime = file.mimetype;
  const b64 = file.buffer.toString("base64");
  if (mime === "application/pdf") {
    return {
      type: "document",
      source: { type: "base64", media_type: "application/pdf", data: b64 },
    };
  }
  if (["image/png", "image/jpeg", "image/gif", "image/webp"].includes(mime)) {
    return {
      type: "image",
      source: { type: "base64", media_type: mime, data: b64 },
    };
  }
  if (mime === "text/plain" || file.originalname.match(/\.(txt|md)$/i)) {
    return {
      type: "document",
      source: {
        type: "text",
        media_type: "text/plain",
        data: file.buffer.toString("utf-8"),
      },
    };
  }
  return null;
}

app.use(express.static(path.join(here, "public")));
app.use("/samples", express.static(path.join(here, "docs", "samples")));
// 転記先サンプルページ(docs/と共用)
app.use("/kessai.html", express.static(path.join(here, "docs", "kessai.html")));
app.use("/ledger.html", express.static(path.join(here, "docs", "ledger.html")));

app.post("/api/extract", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "ファイルがありません" });
    }
    const block = fileToContentBlock(req.file);
    if (!block) {
      return res.status(400).json({
        error: `未対応のファイル形式です: ${req.file.mimetype}(PDF / PNG / JPEG / GIF / WebP / テキストに対応)`,
      });
    }

    // フォーマット指定があればカスタム抽出、なければデフォルト13項目
    const formatText = (req.body?.format || "").trim();
    const custom = !!formatText;

    const response = await client.messages.create({
      model: MODEL,
      max_tokens: 4096,
      system: SYSTEM_PROMPT,
      output_config: {
        format: {
          type: "json_schema",
          schema: custom ? ITEMS_SCHEMA : EXTRACTION_SCHEMA,
        },
      },
      messages: [
        {
          role: "user",
          content: [
            block,
            {
              type: "text",
              text: custom
                ? `次のフォーマット(抽出したい項目の定義)に従って、この契約書から情報を抽出してください。
- フォーマットに書かれた各項目を、書かれた順番どおりにitemsとして返す
- labelには項目名を原文のまま入れる
- 書面から読み取れない項目のvalueはnullにする(推測で値を作らない)
- 金額・日付などの表記ルールはフォーマット内の指示があればそれに従う

【フォーマット】
${formatText}`
                : "この契約書から経理処理に必要な情報を抽出してください。",
            },
          ],
        },
      ],
    });

    if (response.stop_reason === "refusal") {
      return res
        .status(422)
        .json({ error: "この文書は処理できませんでした(安全上の理由)" });
    }
    if (response.stop_reason === "max_tokens") {
      return res
        .status(422)
        .json({ error: "出力が長すぎて途中で切れました。再試行してください" });
    }

    const text = response.content.find((b) => b.type === "text")?.text;
    if (!text) {
      return res.status(500).json({ error: "抽出結果を取得できませんでした" });
    }

    const parsed = JSON.parse(text);
    if (custom) {
      res.json({ filename: req.file.originalname, mode: "custom", items: parsed.items });
    } else {
      res.json({ filename: req.file.originalname, mode: "default", data: parsed });
    }
  } catch (err) {
    if (err instanceof Anthropic.AuthenticationError) {
      return res.status(500).json({
        error:
          "APIキーが無効です。環境変数 ANTHROPIC_API_KEY を確認してください",
      });
    }
    if (err instanceof Anthropic.RateLimitError) {
      return res
        .status(429)
        .json({ error: "APIのレート制限に達しました。少し待って再試行してください" });
    }
    if (err instanceof Anthropic.APIError) {
      return res
        .status(502)
        .json({ error: `Claude APIエラー (${err.status}): ${err.message}` });
    }
    console.error(err);
    res.status(500).json({ error: "サーバーエラーが発生しました" });
  }
});

// multer等のエラーをJSONで返す
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    const msg =
      err.code === "LIMIT_FILE_SIZE"
        ? "ファイルサイズが上限(30MB)を超えています"
        : `アップロードエラー: ${err.message}`;
    return res.status(400).json({ error: msg });
  }
  console.error(err);
  res.status(500).json({ error: "サーバーエラーが発生しました" });
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`契約書アブストラクター: http://localhost:${port}`);
  if (!process.env.ANTHROPIC_API_KEY) {
    console.warn(
      "警告: ANTHROPIC_API_KEY が設定されていません。抽出APIは失敗します。"
    );
  }
});
