"use client";

/**
 * The dashboard's single state container.
 *
 * Shape of the problem: four independent server resources (overview, ward list,
 * one patient's detail, one LLM narration) with their own load states, plus two
 * pieces of user selection (which patient, which hour) that *drive* fetches.
 * A reducer keeps that honest — every transition is one named action, and the
 * per-resource status flags never drift out of sync with the data they describe.
 *
 * Two rules the reducer enforces, both of which are easy to get wrong ad hoc:
 *
 * 1. Selecting a patient clears `selectedHour`. An hour belongs to a patient;
 *    carrying it across would request an hour the new patient may not have.
 * 2. Refetching keeps the previous payload in place and only flips the status,
 *    so the UI dims rather than collapsing to a skeleton (no layout jump).
 */

import { useCallback, useEffect, useMemo, useReducer } from "react";

import { ApiError, api } from "@/lib/api";
import type { Narration, Overview, PatientDetail, PatientSummary } from "@/lib/types";

export type LoadStatus = "idle" | "loading" | "ready" | "error";

export interface DashboardState {
  overview: Overview | null;
  patients: PatientSummary[];
  detail: PatientDetail | null;
  narration: Narration | null;

  selectedPatientId: string | null;
  /** null = the patient's latest window; a number pins one point on the timeline. */
  selectedHour: number | null;

  status: Record<"overview" | "patients" | "detail" | "narration", LoadStatus>;
  errors: Partial<Record<"overview" | "patients" | "detail" | "narration", string>>;
}

type Action =
  | { type: "load"; resource: keyof DashboardState["status"] }
  | { type: "overview/ok"; data: Overview }
  | { type: "patients/ok"; data: PatientSummary[] }
  | { type: "detail/ok"; data: PatientDetail }
  | { type: "narration/ok"; data: Narration }
  | { type: "fail"; resource: keyof DashboardState["status"]; message: string }
  | { type: "selectPatient"; patientId: string }
  | { type: "selectHour"; hour: number | null };

const initialState: DashboardState = {
  overview: null,
  patients: [],
  detail: null,
  narration: null,
  selectedPatientId: null,
  selectedHour: null,
  status: { overview: "idle", patients: "idle", detail: "idle", narration: "idle" },
  errors: {},
};

function reducer(state: DashboardState, action: Action): DashboardState {
  switch (action.type) {
    case "load":
      // Payload is deliberately left in place — see rule 2 above.
      return {
        ...state,
        status: { ...state.status, [action.resource]: "loading" },
        errors: { ...state.errors, [action.resource]: undefined },
      };

    case "overview/ok":
      return { ...state, overview: action.data, status: { ...state.status, overview: "ready" } };

    case "patients/ok":
      return {
        ...state,
        patients: action.data,
        status: { ...state.status, patients: "ready" },
        // Land on the highest-risk patient so the view is never empty on arrival.
        selectedPatientId: state.selectedPatientId ?? action.data[0]?.patient_id ?? null,
      };

    case "detail/ok":
      return { ...state, detail: action.data, status: { ...state.status, detail: "ready" } };

    case "narration/ok":
      return { ...state, narration: action.data, status: { ...state.status, narration: "ready" } };

    case "fail":
      return {
        ...state,
        status: { ...state.status, [action.resource]: "error" },
        errors: { ...state.errors, [action.resource]: action.message },
      };

    case "selectPatient":
      if (action.patientId === state.selectedPatientId) return state;
      return {
        ...state,
        selectedPatientId: action.patientId,
        selectedHour: null, // rule 1
        narration: null,
        status: { ...state.status, narration: "idle" },
        errors: { ...state.errors, narration: undefined },
      };

    case "selectHour":
      if (action.hour === state.selectedHour) return state;
      return {
        ...state,
        selectedHour: action.hour,
        narration: null,
        status: { ...state.status, narration: "idle" },
        errors: { ...state.errors, narration: undefined },
      };

    default:
      return state;
  }
}

const message = (error: unknown) =>
  error instanceof ApiError || error instanceof Error ? error.message : "알 수 없는 오류";

const aborted = (error: unknown) => error instanceof DOMException && error.name === "AbortError";

export function useDashboard(patientLimit = 40) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const { selectedPatientId, selectedHour } = state;

  useEffect(() => {
    const controller = new AbortController();
    dispatch({ type: "load", resource: "overview" });
    dispatch({ type: "load", resource: "patients" });

    api
      .overview(controller.signal)
      .then((data) => dispatch({ type: "overview/ok", data }))
      .catch((error) => {
        if (!aborted(error)) dispatch({ type: "fail", resource: "overview", message: message(error) });
      });

    api
      .patients(patientLimit, controller.signal)
      .then((data) => dispatch({ type: "patients/ok", data }))
      .catch((error) => {
        if (!aborted(error)) dispatch({ type: "fail", resource: "patients", message: message(error) });
      });

    return () => controller.abort();
  }, [patientLimit]);

  useEffect(() => {
    if (!selectedPatientId) return;
    const controller = new AbortController();
    dispatch({ type: "load", resource: "detail" });

    api
      .patient(selectedPatientId, selectedHour, controller.signal)
      .then((data) => dispatch({ type: "detail/ok", data }))
      .catch((error) => {
        if (!aborted(error)) dispatch({ type: "fail", resource: "detail", message: message(error) });
      });

    return () => controller.abort();
  }, [selectedPatientId, selectedHour]);

  const selectPatient = useCallback(
    (patientId: string) => dispatch({ type: "selectPatient", patientId }),
    [],
  );
  const selectHour = useCallback((hour: number | null) => dispatch({ type: "selectHour", hour }), []);

  /** Explicit, never automatic: the LLM call costs a request, so a click triggers it. */
  const evidence = state.detail?.evidence ?? null;
  const explain = useCallback(async () => {
    if (!evidence) return;
    dispatch({ type: "load", resource: "narration" });
    try {
      const data = await api.explain(evidence);
      dispatch({ type: "narration/ok", data });
    } catch (error) {
      dispatch({ type: "fail", resource: "narration", message: message(error) });
    }
  }, [evidence]);

  const selectedPatient = useMemo(
    () => state.patients.find((patient) => patient.patient_id === selectedPatientId) ?? null,
    [state.patients, selectedPatientId],
  );

  return { ...state, selectedPatient, selectPatient, selectHour, explain };
}
