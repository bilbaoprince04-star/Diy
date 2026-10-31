import { createOpenAI } from "@ai-sdk/openai";
import { streamText, type ModelMessage } from "ai";
import { z } from "zod";

import {
  createLovableAiGatewayRunIdFetch,
  getLovableAiGatewayRunId,
  withLovableAiGatewayRunIdHeader,
} from "./run-id.server";

const GATEWAY = "https://ai.gateway.lovable.dev/v1";
const MODEL = "openai/gpt-6-astra";

const attachmentSchema = z.object({
  name: z.string().max(300),
  kind: z.enum(["image", "text"]),
  mime: z.string().max(120),
  content: z.string().max(6_000_000),
});

const turnSchema = z.object({
  role: z.enum(["user", "assistant"]),
  content: z.string().max(400_000),
});

const bodySchema = z.object({
  prompt: z.string().min(1).max(40_000),
  framework: z.enum(["html", "tailwind", "react"]),
  behavior: z.string().max(4000).optional().default(""),
  currentCode: z.string().max(400_000).optional().default(""),
  history: z.array(turnSchema).max(20).optional().default([]),
  attachments: z.array(attachmentSchema).max(12).optional().default([]),
});

function frameworkRules(fw: "html" | "tailwind" | "react") {
  if (fw === "html")
    return "Use semantic HTML5, a <style> block with modern CSS, and vanilla JavaScript in a <script> block. No external CSS frameworks.";
  if (fw === "react")
    return 'Use React 18 via UMD (<script src="https://unpkg.com/react@18/umd/react.production.min.js">, react-dom likewise) and Babel standalone (<script src="https://unpkg.com/@babel/standalone/babel.min.js">) with <script type="text/babel">. Style with the Tailwind Play CDN (<script src="https://cdn.tailwindcss.com"></script>). Write clean functional components.';
  return 'Style with the Tailwind Play CDN (<script src="https://cdn.tailwindcss.com"></script>) and vanilla JavaScript.';
}

export async function handleGenerate(request: Request) {
  let body: z.infer<typeof bodySchema>;
  try {
    body = bodySchema.parse(await request.json());
  } catch {
    return Response.json({ error: "Invalid request. Check your prompt and attachments." }, { status: 400 });
  }

  const apiKey = process.env["LOVABLE_API_KEY"];
  if (!apiKey) return Response.json({ error: "AI is not configured." }, { status: 500 });

  const instructions = [
    "You are VIPER, an elite website-building AI. You produce complete, production-quality, responsive websites.",
    "ALWAYS reply with a brief 1-3 sentence summary of what you built or changed, then ONE fenced code block ```html containing a COMPLETE self-contained HTML document (<!DOCTYPE html> ... </html>).",
    "Multi-page requests: implement every page inside the single document with client-side section switching (hash navigation).",
    frameworkRules(body.framework),
    "Never use external image URLs that may break; use inline SVG, CSS gradients, or emoji for visuals.",
    "Design quality bar: a distinctive visual direction, strong typography hierarchy, generous spacing, polished hover/focus states, subtle motion, real-sounding copy (no lorem ipsum), fully responsive from 360px up, accessible contrast and labels.",
    "Double-check your code for syntax errors before answering; it must run as-is in a sandboxed iframe.",
    "If the user reports runtime errors, find the root cause and fix it without removing features.",
    "If the user attached screenshots, reproduce their layout and style closely. If they attached code or config files (.lua, .js, .json, etc.), use them as the source of truth for content, data, or logic.",
    body.currentCode
      ? "The user is iterating on an existing site. Apply the requested changes to the CURRENT CODE and return the full updated document."
      : "",
    body.behavior ? `Extra user preferences: ${body.behavior}` : "",
  ]
    .filter(Boolean)
    .join("\n");

  const parts: Array<
    { type: "text"; text: string } | { type: "image"; image: string; mediaType?: string }
  > = [];
  for (const a of body.attachments) {
    if (a.kind === "image") {
      parts.push({ type: "text", text: `Attached image: ${a.name}` });
      parts.push({ type: "image", image: a.content, mediaType: a.mime });
    } else {
      parts.push({ type: "text", text: `Attached file "${a.name}":\n\`\`\`\n${a.content}\n\`\`\`` });
    }
  }
  if (body.currentCode) parts.push({ type: "text", text: `CURRENT CODE:\n\`\`\`html\n${body.currentCode}\n\`\`\`` });
  parts.push({ type: "text", text: body.prompt });

  const messages: ModelMessage[] = [
    ...body.history.map((t) => ({ role: t.role, content: t.content }) as ModelMessage),
    { role: "user", content: parts },
  ];

  const runIdFetch = createLovableAiGatewayRunIdFetch(getLovableAiGatewayRunId(request));
  const provider = createOpenAI({
    baseURL: GATEWAY,
    apiKey,
    headers: { "Lovable-API-Key": apiKey, "X-Lovable-AIG-SDK": "vercel-ai-sdk" },
    fetch: runIdFetch.fetch,
  });

  let upstreamError: { status: number; message: string } | null = null;
  const result = streamText({
    model: provider.responses(MODEL),
    instructions,
    messages,
    abortSignal: request.signal,
    maxRetries: 0,
    providerOptions: {
      openai: {
        store: false,
        forceReasoning: true,
        reasoningEffort: "medium",
        reasoningSummary: "detailed",
        include: ["reasoning.encrypted_content"],
      },
    },
    onError: ({ error }) => {
      const e = error as { statusCode?: number; message?: string };
      upstreamError = { status: e.statusCode ?? 500, message: e.message ?? "Generation failed" };
      console.error("VIPER generation error", e.statusCode, e.message);
    },
  });

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        // Reasoning ("thinking") deltas are wrapped in \u0002…\u0003 so the UI can show them separately.
        for await (const part of result.fullStream) {
          if (part.type === "text-delta") controller.enqueue(encoder.encode(part.text));
          else if (part.type === "reasoning-delta" && part.text)
            controller.enqueue(encoder.encode(`\u0002${part.text.replace(/[\u0002\u0003]/g, "")}\u0003`));
        }
        if (upstreamError) {
          const err = upstreamError as { status: number; message: string };
          const msg =
            err.status === 402
              ? "AI credits are used up for this workspace. Add credits to keep generating."
              : err.status === 429
                ? "Too many requests right now. Wait a moment and try again."
                : "Generation failed. Please try again.";
          controller.enqueue(encoder.encode(`\n[[VIPER_ERROR:${msg}]]`));
        }
        controller.close();
      } catch (error) {
        if (request.signal.aborted) controller.close();
        else {
          controller.enqueue(encoder.encode("\n[[VIPER_ERROR:Generation failed. Please try again.]]"));
          controller.close();
          console.error(error);
        }
      }
    },
  });

  return withLovableAiGatewayRunIdHeader(
    new Response(stream, { headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-cache, no-transform" } }),
    runIdFetch,
    undefined,
    { waitForUpstreamHeaders: false },
  );
}
