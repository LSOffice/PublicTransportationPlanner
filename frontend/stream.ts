export interface GenerationEvent {
  id: number;
  type: string;
  message: string;
  plan?: unknown;
}
export function decodeGenerationEvent(text: string): GenerationEvent {
  const data = JSON.parse(text);
  if (!Number.isSafeInteger(data.id) || data.id < 1 || typeof data.type !== "string" || typeof data.message !== "string") {
    throw new Error("Malformed generation event");
  }
  return data;
}
