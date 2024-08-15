import * as webllm from "@mlc-ai/web-llm";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import type { Document } from "@langchain/core/documents";
import { formatDocumentsAsString } from "langchain/util/document";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

const initProgressCallback = (report: webllm.InitProgressReport) => {
  setLabel("init-label", report.text);
};

// For integration with Langchain
class WebLLMEmbeddings implements EmbeddingsInterface {
  engine: webllm.MLCEngineInterface;
  modelId: string;
  constructor(engine: webllm.MLCEngineInterface, modelId: string) {
    this.engine = engine;
    this.modelId = modelId;
  }

  async _embed(texts: string[]): Promise<number[][]> {
    const reply = await this.engine.embeddings.create({
      input: texts,
      model: this.modelId,
    });
    const result: number[][] = [];
    for (let i = 0; i < texts.length; i++) {
      result.push(reply.data[i].embedding);
    }
    return result;
  }

  async embedQuery(document: string): Promise<number[]> {
    return this._embed([document]).then((embeddings) => embeddings[0]);
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    return this._embed(documents);
  }
}

// Prepare inputs
const documents_og = ["The Data Cloud!", "Mexico City of Course!"];
const queries_og = ["what is snowflake?", "Where can I get the best tacos?"];
const documents: string[] = [];
const queries: string[] = [];
const query_prefix =
  "Represent this sentence for searching relevant passages: ";
// Process according to Snowflake model
documents_og.forEach(function (item, index) {
  documents[index] = `[CLS] ${item} [SEP]`;
});
queries_og.forEach(function (item, index) {
  queries[index] = `[CLS] ${query_prefix}${item} [SEP]`;
});
console.log("Formatted documents: ", documents);
console.log("Formatted queries: ", queries);

// Using webllm's API
async function webllmAPI() {
  // b4 means the max batch size is compiled as 4. That is, the model can process 4 inputs in a
  // batch. If given more than 4, the model will forward multiple times. The larger the max batch
  // size, the more memory it consumes.
  // const selectedModel = "snowflake-arctic-embed-m-q0f32-MLC-b32";
  const selectedModel = "snowflake-arctic-embed-m-q0f32-MLC-b4";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    {
      initProgressCallback: initProgressCallback,
      logLevel: "INFO", // specify the log level
    },
  );

  const docReply = await engine.embeddings.create({ input: documents });
  console.log(docReply);
  console.log(docReply.usage);

  const queryReply = await engine.embeddings.create({ input: queries });
  console.log(queryReply);
  console.log(queryReply.usage);

  // Calculate similarity (we use langchain here, but any method works)
  const vectorStore = await MemoryVectorStore.fromExistingIndex(
    new WebLLMEmbeddings(engine, selectedModel),
  );
  // See score
  for (let i = 0; i < queries_og.length; i++) {
    console.log(`Similarity with: ${queries_og[i]}`);
    for (let j = 0; j < documents_og.length; j++) {
      const similarity = vectorStore.similarity(
        queryReply.data[i].embedding,
        docReply.data[j].embedding,
      );
      console.log(`${documents_og[j]}: ${similarity}`);
    }
  }
}

// Alternatively, integrating with Langchain's API
async function langchainAPI() {
  // b4 means the max batch size is compiled as 4. That is, the model can process 4 inputs in a
  // batch. If given more than 4, the model will forward multiple times. The larger the max batch
  // size, the more memory it consumes.
  // const selectedModel = "snowflake-arctic-embed-m-q0f32-MLC-b32";
  const selectedModel = "snowflake-arctic-embed-m-q0f32-MLC-b4";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    {
      initProgressCallback: initProgressCallback,
      logLevel: "INFO", // specify the log level
    },
  );

  const vectorStore = await MemoryVectorStore.fromExistingIndex(
    new WebLLMEmbeddings(engine, selectedModel),
  );
  const document0: Document = {
    pageContent: documents[0],
    metadata: {},
  };
  const document1: Document = {
    pageContent: documents[1],
    metadata: {},
  };
  await vectorStore.addDocuments([document0, document1]);

  const similaritySearchResults0 = await vectorStore.similaritySearch(
    queries[0],
    1,
  );
  for (const doc of similaritySearchResults0) {
    console.log(`* ${doc.pageContent}`);
  }

  const similaritySearchResults1 = await vectorStore.similaritySearch(
    queries[1],
    1,
  );
  for (const doc of similaritySearchResults1) {
    console.log(`* ${doc.pageContent}`);
  }
}

// RAG with Langchain.js using WebLLM for both LLM and Embedding in a single engine
// Followed https://js.langchain.com/v0.1/docs/expression_language/cookbook/retrieval/
// There are many possible ways to achieve RAG (e.g. degree of integration with Langchain,
// using WebWorker, etc.). We provide a minimal example here.
async function simpleRAG() {
  // 0. Load both embedding model and LLM to a single WebLLM Engine
  const embeddingModelId = "snowflake-arctic-embed-m-q0f32-MLC-b4";
  const llmModelId = "gemma-2-2b-it-q4f32_1-MLC-1k";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    [embeddingModelId, llmModelId],
    {
      initProgressCallback: initProgressCallback,
      logLevel: "INFO", // specify the log level
    },
  );

  const vectorStore = await MemoryVectorStore.fromTexts(
    ["mitochondria is the powerhouse of the cell"],
    [{ id: 1 }],
    new WebLLMEmbeddings(engine, embeddingModelId),
  );
  const retriever = vectorStore.asRetriever();

  const prompt =
    PromptTemplate.fromTemplate(`Answer the question based only on the following context:
  {context}
  
  Question: {question}`);

  const chain = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocumentsAsString),
      question: new RunnablePassthrough(),
    },
    prompt,
  ]);

  const formattedPrompt = (
    await chain.invoke("What is the powerhouse of the cell?")
  ).toString();
  const reply = await engine.chat.completions.create({
    messages: [{ role: "user", content: formattedPrompt }],
    model: llmModelId,
  });

  console.log(reply.choices[0].message.content);

  /*
    "The powerhouse of the cell is the mitochondria."
  */
}

// Select one to run
// webllmAPI();
// langchainAPI();
simpleRAG();
