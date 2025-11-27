/**
 * API client for KohakuPaper backend
 */

import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  timeout: 30000,
});

/**
 * @typedef {Object} Paper
 * @property {string} id
 * @property {string} title
 * @property {string} status
 * @property {string} abstract
 * @property {string} primary_area
 * @property {string} author
 * @property {number} rating_avg
 * @property {number} confidence_avg
 * @property {string} site
 */

/**
 * @typedef {Object} PapersResponse
 * @property {Paper[]} papers
 * @property {number} total
 * @property {number} limit
 * @property {number} offset
 */

/**
 * Fetch papers with filters
 * @param {Object} params
 * @returns {Promise<PapersResponse>}
 */
export async function fetchPapers(params = {}) {
  const { data } = await api.get("/papers", { params });
  return data;
}

/**
 * Fetch available filters
 * @param {Object} params
 * @returns {Promise<Object>}
 */
export async function fetchFilters(params = {}) {
  const { data } = await api.get("/papers/filters", { params });
  return data;
}

/**
 * Fetch statistics summary
 * @param {Object} params
 * @returns {Promise<Object>}
 */
export async function fetchSummary(params = {}) {
  const { data } = await api.get("/statistics/summary", { params });
  return data;
}

/**
 * Fetch statistics by status
 * @param {Object} params
 * @returns {Promise<Array>}
 */
export async function fetchStatsByStatus(params = {}) {
  const { data } = await api.get("/statistics/by-status", { params });
  return data;
}

/**
 * Fetch statistics by area
 * @param {Object} params
 * @returns {Promise<Array>}
 */
export async function fetchStatsByArea(params = {}) {
  const { data } = await api.get("/statistics/by-area", { params });
  return data;
}

/**
 * Fetch rating distribution
 * @param {Object} params
 * @returns {Promise<Object>}
 */
export async function fetchRatingDistribution(params = {}) {
  const { data } = await api.get("/statistics/rating-distribution", { params });
  return data;
}

/**
 * Fetch yearly statistics
 * @param {Object} params
 * @returns {Promise<Array>}
 */
export async function fetchYearlyStats(params = {}) {
  const { data } = await api.get("/statistics/yearly", { params });
  return data;
}

/**
 * Fetch top areas
 * @param {Object} params
 * @returns {Promise<Array>}
 */
export async function fetchTopAreas(params = {}) {
  const { data } = await api.get("/statistics/top-areas", { params });
  return data;
}

/**
 * Fetch conferences list (legacy)
 * @returns {Promise<Object>}
 */
export async function fetchConferences() {
  const { data } = await api.get("/data/conferences");
  return data;
}

/**
 * Fetch local files (legacy)
 * @returns {Promise<Object>}
 */
export async function fetchLocalFiles() {
  const { data } = await api.get("/data/local-files");
  return data;
}

/**
 * Download conference data (legacy)
 * @param {string} conference
 * @param {number} [year]
 * @returns {Promise<Object>}
 */
export async function downloadConference(conference, year = null) {
  const params = year ? { year } : {};
  const { data } = await api.post(`/data/download/${conference}`, null, {
    params,
  });
  return data;
}

// ============ Sync API ============

/**
 * Fetch all available conferences from papercopilot with download status
 * @returns {Promise<{conferences: Array, local_count: number}>}
 */
export async function fetchAvailableConferences() {
  const { data } = await api.get("/sync/available");
  return data;
}

/**
 * Fetch locally downloaded conferences
 * @returns {Promise<Array>}
 */
export async function fetchLocalConferences() {
  const { data } = await api.get("/sync/local");
  return data;
}

/**
 * Get sync status
 * @returns {Promise<Object>}
 */
export async function fetchSyncStatus() {
  const { data } = await api.get("/sync/status");
  return data;
}

/**
 * Update the paperlists repository
 * @returns {Promise<{success: boolean}>}
 */
export async function updateRepo() {
  const { data } = await api.post("/sync/update-repo");
  return data;
}

/**
 * Sync a specific conference (download and compute diffs)
 * NO git clone required - uses GitHub API for everything!
 * @param {string} conference
 * @param {number} year
 * @returns {Promise<{success: boolean, message: string}>}
 */
export async function syncConference(conference, year) {
  const { data } = await api.post("/sync/conference", null, {
    params: { conference, year },
    timeout: 300000, // 5 minutes for large conferences
  });
  return data;
}

export default api;
