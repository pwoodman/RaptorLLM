/** Rough token count (about four characters per token). This is not the model's tokenizer. */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

export function estimateTokenTotal(texts: readonly string[]): number {
  return texts.reduce((sum, text) => sum + estimateTokens(text), 0);
}
