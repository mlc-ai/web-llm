/** 
 * Global configuration for anticipatory infilling models.
 * 
 * From https://github.com/jthickstun/anticipation/blob/main/anticipation/sample.py.
*/

export const CONTEXT_SIZE = 1024                // model context
export const EVENT_SIZE = 3                     // each event/control is encoded as 3 tokens
export const M = 341                            // model context (1024 = 1 + EVENT_SIZE*M)
export const DELTA = 5                          // anticipation time in seconds

if (CONTEXT_SIZE != 1 + EVENT_SIZE * M) throw Error

// vocabulary constants

export const MAX_TIME_IN_SECONDS = 100          // exclude very long training sequences
export const MAX_DURATION_IN_SECONDS = 10       // maximum duration of a note
export const TIME_RESOLUTION = 100              // 10ms time resolution = 100 bins/second

export const MAX_PITCH = 128                    // 128 MIDI pitches
export const MAX_INSTR = 129                    // 129 MIDI instruments (128 + drums)
export const MAX_NOTE = MAX_PITCH * MAX_INSTR     // note = pitch x instrument

export const MAX_INTERARRIVAL_IN_SECONDS = 10   // maximum interarrival time (for MIDI-like encoding)

// preprocessing settings

export const PREPROC_WORKERS = 16

export const COMPOUND_SIZE = 5                  // event size in the intermediate compound tokenization
export const MAX_TRACK_INSTR = 16               // exclude tracks with large numbers of instruments
export const MAX_TRACK_TIME_IN_SECONDS = 3600   // exclude very long tracks (longer than 1 hour)
export const MIN_TRACK_TIME_IN_SECONDS = 10     // exclude very short tracks (less than 10 seconds)
export const MIN_TRACK_EVENTS = 100             // exclude very short tracks (less than 100 events)

// LakhMIDI dataset splits

export const LAKH_SPLITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
export const LAKH_VALID = ['e']
export const LAKH_TEST = ['f']

// derived quantities

export const MAX_TIME = TIME_RESOLUTION * MAX_TIME_IN_SECONDS
export const MAX_DUR = TIME_RESOLUTION * MAX_DURATION_IN_SECONDS

export const MAX_INTERARRIVAL = TIME_RESOLUTION * MAX_INTERARRIVAL_IN_SECONDS