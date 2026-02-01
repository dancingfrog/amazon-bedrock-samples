import { createAgent, tool } from "langchain";
import { z } from "zod";

const getWeather = tool((input) => `It's always sunny in ${input.city}!`, {
  name: "get_weather",
  description: "Get the weather for a given city",
  schema: z.object({
    city: z.string().describe("The city to get the weather for"),
  }),
});

const agent = createAgent({
  model: "anthropic:claude-sonnet-4-5-20250929",
  tools: [getWeather],
});

// Run the agent
const result = await agent.invoke({
  messages: [{ role: "user", content: "What's the weather in San Francisco?" }],
});

console.log(result);

// =============================================================================
// LangGraph + Amazon Bedrock Example
// =============================================================================

import {
  BedrockRuntimeClient,
  ConverseCommand,
  type Message,
  type ContentBlock,
  type ToolConfiguration,
  type ToolResultContentBlock,
} from "@aws-sdk/client-bedrock-runtime";
import { StateGraph, Annotation } from "@langchain/langgraph";

// Initialize Bedrock client
const bedrockClient = new BedrockRuntimeClient({ region: "us-east-1" });
const BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0";

// Define the graph state using LangGraph's Annotation
const GraphState = Annotation.Root({
  messages: Annotation<Message[]>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),
});

// Define the tool schema for Bedrock
const bedrockToolConfig: ToolConfiguration = {
  tools: [
    {
      toolSpec: {
        name: "search",
        description: "Use this tool to query the web for weather information.",
        inputSchema: {
          json: {
            type: "object",
            properties: {
              query: {
                type: "string",
                description: "The search query",
              },
            },
            required: ["query"],
          },
        },
      },
    },
  ],
};

// Tool implementation
const searchTool = async ({ query }: { query: string }): Promise<string> => {
  console.log(`[Search Tool] Querying: ${query}`);
  if (
    query.toLowerCase().includes("sf") ||
    query.toLowerCase().includes("san francisco")
  ) {
    return "It's 60 degrees and foggy in San Francisco.";
  }
  return "It's 90 degrees and sunny.";
};

// Node: Call Bedrock model
const callBedrockModel = async (
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> => {
  console.log("[Model] Calling Bedrock...");

  const command = new ConverseCommand({
    modelId: BEDROCK_MODEL_ID,
    messages: state.messages,
    toolConfig: bedrockToolConfig,
  });

  const response = await bedrockClient.send(command);

  const assistantMessage: Message = {
    role: "assistant",
    content: response.output?.message?.content ?? [],
  };

  console.log("[Model] Stop reason:", response.stopReason);
  return { messages: [assistantMessage] };
};

// Node: Execute tools
const executeTools = async (
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> => {
  const lastMessage = state.messages[state.messages.length - 1];
  const toolUseBlocks =
    lastMessage.content?.filter(
      (block: ContentBlock): block is ContentBlock.ToolUseMember =>
        "toolUse" in block
    ) ?? [];

  if (toolUseBlocks.length === 0) {
    throw new Error("No tool calls found in message.");
  }

  const toolResults: Message = {
    role: "user",
    content: [],
  };

  for (const toolBlock of toolUseBlocks) {
    const toolUse = toolBlock.toolUse!;
    console.log(`[Tools] Executing: ${toolUse.name}`);

    let resultContent: string;
    if (toolUse.name === "search") {
      resultContent = await searchTool(toolUse.input as { query: string });
    } else {
      resultContent = `Unknown tool: ${toolUse.name}`;
    }

    const toolResultBlock: ToolResultContentBlock = {
      toolUseId: toolUse.toolUseId!,
      content: [{ text: resultContent }],
    };

    (toolResults.content as any[]).push({ toolResult: toolResultBlock });
  }

  return { messages: [toolResults] };
};

// Conditional edge: Check if we should continue to tools or end
const shouldContinue = (
  state: typeof GraphState.State
): "tools" | "__end__" => {
  const lastMessage = state.messages[state.messages.length - 1];
  const hasToolUse = lastMessage.content?.some(
    (block: ContentBlock) => "toolUse" in block
  );

  if (hasToolUse) {
    return "tools";
  }
  return "__end__";
};

// Build the graph
const bedrockWorkflow = new StateGraph(GraphState)
  .addNode("model", callBedrockModel)
  .addNode("tools", executeTools)
  .addEdge("__start__", "model")
  .addConditionalEdges("model", shouldContinue, {
    tools: "tools",
    __end__: "__end__",
  })
  .addEdge("tools", "model")
  .compile();

// Run the Bedrock LangGraph example
console.log("\n" + "=".repeat(60));
console.log("Running LangGraph + Amazon Bedrock Example");
console.log("=".repeat(60) + "\n");

const bedrockResult = await bedrockWorkflow.invoke({
  messages: [
    {
      role: "user",
      content: [{ text: "What is the weather in Los Angeles?" }],
    },
  ],
});

// Extract and display the final response
const finalMessage = bedrockResult.messages[bedrockResult.messages.length - 1];
const finalText = finalMessage.content
  ?.filter(
    (block: ContentBlock): block is ContentBlock.TextMember => "text" in block
  )
  .map((block: ContentBlock.TextMember) => block.text)
  .join("\n");

console.log("\n[Final Response]:", finalText);
