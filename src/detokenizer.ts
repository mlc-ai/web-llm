/**
 * Decodes a growing token sequence while decoding only a few of the newest
 * tokens per step, using the prefix/read offsets of HF tokenizers'
 * `DecodeStream` and vLLM's `detokenize_incrementally`.
 *
 * Text before `readOffset` is final. Each push decodes the tokens from
 * `prefixOffset` on and keeps what follows the decoded text of
 * [prefixOffset, readOffset), so a decoder that depends on the preceding token
 * (leading-space strip, byte fallback, a character split across tokens) sees
 * the context it sees in a full decode. The offsets advance only when the
 * window does not end in U+FFFD, i.e. no character is still incomplete. If a
 * token changes text before the window, the window no longer starts with that
 * prefix text and the whole sequence is decoded instead.
 */
export class IncrementalDetokenizer {
  private ids: number[] = [];
  private prefixOffset = 0;
  private readOffset = 0;
  // decode(ids[0, readOffset)) and decode(ids[prefixOffset, readOffset)).
  private readText = "";
  private prefixText = "";
  // False after a full decode that rewrote text before readOffset.
  private textHasReadText = true;
  /** decode(ids). */
  text = "";
  /** Length of the text the last push left unchanged; 0 after a full decode. */
  unchangedLength = 0;

  constructor(private readonly decode: (ids: Int32Array) => string) {}

  reset(): void {
    this.ids = [];
    this.prefixOffset = 0;
    this.readOffset = 0;
    this.readText = "";
    this.prefixText = "";
    this.textHasReadText = true;
    this.text = "";
    this.unchangedLength = 0;
  }

  push(id: number): string {
    this.ids.push(id);
    const windowText = this.decodeFrom(this.prefixOffset);
    let grew = true;
    if (windowText.startsWith(this.prefixText)) {
      this.unchangedLength = this.textHasReadText ? this.readText.length : 0;
      this.text = this.readText + windowText.slice(this.prefixText.length);
      this.textHasReadText = true;
      grew = windowText.length > this.prefixText.length;
    } else {
      this.unchangedLength = 0;
      this.text = this.decodeFrom(0);
      this.textHasReadText = false;
    }
    if (grew && !this.text.endsWith("\ufffd")) {
      this.readText = this.text;
      this.textHasReadText = true;
      this.prefixOffset = this.readOffset;
      this.readOffset = this.ids.length;
      this.prefixText = this.decodeFrom(this.prefixOffset, this.readOffset);
      // Text that decodes to nothing (a stripped leading space) would match
      // any window, so widen the prefix until it decodes to some text.
      while (this.prefixText === "" && this.prefixOffset > 0) {
        this.prefixText = this.decodeFrom(--this.prefixOffset, this.readOffset);
      }
    }
    return this.text;
  }

  private decodeFrom(begin: number, end = this.ids.length): string {
    return this.decode(Int32Array.from(this.ids.slice(begin, end)));
  }
}
