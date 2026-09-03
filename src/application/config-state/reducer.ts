import type { ParapperConfig } from "../../lib/types";

export type ConfigState = {
  current: ParapperConfig | null;
  applied: ParapperConfig | null;
  revision: number;
};

export type ConfigAction =
  | { type: "loaded"; config: ParapperConfig }
  | { type: "currentReplaced"; config: ParapperConfig | null }
  | { type: "appliedReplaced"; config: ParapperConfig | null }
  | { type: "optimisticUpdate"; config: ParapperConfig; revision: number }
  | { type: "saveCompleted"; config: ParapperConfig; revision: number }
  | { type: "saveFailed"; revision: number };

export const initialConfigState: ConfigState = {
  current: null,
  applied: null,
  revision: 0,
};

export const configStateReducer = (
  state: ConfigState,
  action: ConfigAction,
): ConfigState => {
  switch (action.type) {
    case "loaded":
      return { ...state, current: action.config, applied: action.config };
    case "currentReplaced":
      return { ...state, current: action.config };
    case "appliedReplaced":
      return { ...state, applied: action.config };
    case "optimisticUpdate":
      return {
        ...state,
        current: action.config,
        revision: action.revision,
      };
    case "saveCompleted":
      if (action.revision !== state.revision) {
        return { ...state, applied: action.config };
      }
      return {
        ...state,
        current: action.config,
        applied: action.config,
      };
    case "saveFailed":
      if (action.revision !== state.revision) return state;
      return { ...state, current: state.applied };
  }
};
