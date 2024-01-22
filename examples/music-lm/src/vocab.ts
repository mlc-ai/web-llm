/**
 * The vocabularies used for arrival-time and interarrival-time encodings.
 * 
 * From https://github.com/jthickstun/anticipation/blob/main/anticipation/sample.py.
 */


// training sequence vocab

import * as config from "./config"

// the event block
export const EVENT_OFFSET = 0
export const TIME_OFFSET = EVENT_OFFSET
export const DUR_OFFSET = TIME_OFFSET + config.MAX_TIME
export const NOTE_OFFSET = DUR_OFFSET + config.MAX_DUR
export const REST = NOTE_OFFSET + config.MAX_NOTE

// the control block
export const CONTROL_OFFSET = NOTE_OFFSET + config.MAX_NOTE + 1
export const ATIME_OFFSET = CONTROL_OFFSET + 0
export const ADUR_OFFSET = ATIME_OFFSET + config.MAX_TIME
export const ANOTE_OFFSET = ADUR_OFFSET + config.MAX_DUR

// the special block
export const SPECIAL_OFFSET = ANOTE_OFFSET + config.MAX_NOTE
export const SEPARATOR = SPECIAL_OFFSET
export const AUTOREGRESS = SPECIAL_OFFSET + 1
export const ANTICIPATE = SPECIAL_OFFSET + 2
export const VOCAB_SIZE = ANTICIPATE + 1

// interarrival - time(MIDI - like) vocab
export const MIDI_TIME_OFFSET = 0
export const MIDI_START_OFFSET = MIDI_TIME_OFFSET + config.MAX_INTERARRIVAL
export const MIDI_END_OFFSET = MIDI_START_OFFSET + config.MAX_NOTE
export const MIDI_SEPARATOR = MIDI_END_OFFSET + config.MAX_NOTE
export const MIDI_VOCAB_SIZE = MIDI_SEPARATOR + 1
