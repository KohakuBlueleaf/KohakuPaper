<script setup>
import { useConferenceStore } from "@/stores/conference";
import {
  fetchSummary,
  fetchAvailableConferences,
  syncConference,
  fetchSyncStatus,
} from "@/utils/api";
import { ElMessage, ElMessageBox } from "element-plus";

const conferenceStore = useConferenceStore();
const router = useRouter();

const summary = ref(null);
const loading = ref(false);

// Available conferences from papercopilot
const availableConferences = ref([]);
const conferencesLoading = ref(false);

// Selected conference/year for download
const selectedDownloadConf = ref("");
const selectedDownloadYear = ref(null);

// Download state
const downloading = ref(false);
const downloadProgress = ref("");

// Sync status
const syncStatus = ref(null);

// Get years for selected conference
const availableYears = computed(() => {
  if (!selectedDownloadConf.value) return [];
  const conf = availableConferences.value.find(
    (c) => c.conference === selectedDownloadConf.value,
  );
  return conf?.years || [];
});

// Check if selected conference/year is already downloaded
const isDownloaded = computed(() => {
  if (!selectedDownloadConf.value || !selectedDownloadYear.value) return false;
  const conf = availableConferences.value.find(
    (c) => c.conference === selectedDownloadConf.value,
  );
  const yearInfo = conf?.years?.find(
    (y) => y.year === selectedDownloadYear.value,
  );
  return yearInfo?.downloaded || false;
});

// Get list of downloaded conferences for sync management
const downloadedConferences = computed(() => {
  const result = [];
  for (const conf of availableConferences.value) {
    for (const yearInfo of conf.years || []) {
      if (yearInfo.downloaded) {
        result.push({
          conference: conf.conference,
          year: yearInfo.year,
          size_mb: yearInfo.size_mb,
        });
      }
    }
  }
  return result.sort((a, b) => {
    if (a.conference !== b.conference)
      return a.conference.localeCompare(b.conference);
    return b.year - a.year;
  });
});

// Load available conferences from papercopilot
async function loadAvailableConferences() {
  conferencesLoading.value = true;
  try {
    const data = await fetchAvailableConferences();
    availableConferences.value = data.conferences || [];
  } catch (err) {
    console.error("Failed to load conferences:", err);
    ElMessage.error("Failed to load available conferences");
  } finally {
    conferencesLoading.value = false;
  }
}

// Download/sync selected conference (no git clone needed!)
async function downloadConference() {
  if (!selectedDownloadConf.value || !selectedDownloadYear.value) {
    ElMessage.warning("Please select a conference and year");
    return;
  }

  downloading.value = true;
  downloadProgress.value = `Syncing ${selectedDownloadConf.value} ${selectedDownloadYear.value}...`;

  try {
    // Sync - downloads file and computes diffs via GitHub API (no git clone!)
    const result = await syncConference(
      selectedDownloadConf.value,
      selectedDownloadYear.value,
    );

    if (result.success) {
      ElMessage.success(
        `Synced ${selectedDownloadConf.value} ${selectedDownloadYear.value}`,
      );
      downloadProgress.value = "Done!";

      // Reload conferences to update download status
      await loadAvailableConferences();

      // Reload store data
      await conferenceStore.loadInitialData();

      // Auto-select the downloaded conference
      conferenceStore.setConference(
        selectedDownloadConf.value,
        selectedDownloadYear.value,
      );
    } else {
      ElMessage.error(result.error || "Sync failed");
      downloadProgress.value = "";
    }
  } catch (err) {
    console.error("Sync error:", err);
    ElMessage.error("Failed to sync conference data");
    downloadProgress.value = "";
  } finally {
    downloading.value = false;
  }
}

// Go to conference page
function goToConference() {
  router.push("/conference");
}

// View selected conference (navigate to conference page)
function viewConference() {
  if (
    selectedDownloadConf.value &&
    selectedDownloadYear.value &&
    isDownloaded.value
  ) {
    conferenceStore.setConference(
      selectedDownloadConf.value,
      selectedDownloadYear.value,
    );
    router.push("/conference");
  }
}

async function loadSummary() {
  if (!conferenceStore.selectedConference) return;

  loading.value = true;
  try {
    summary.value = await fetchSummary({
      conference: conferenceStore.selectedConference,
      year: conferenceStore.selectedYear,
    });
  } catch (err) {
    console.error("Failed to load summary:", err);
  } finally {
    loading.value = false;
  }
}

watch(
  () => [conferenceStore.selectedConference, conferenceStore.selectedYear],
  () => {
    if (conferenceStore.selectedConference) {
      loadSummary();
    }
  },
  { immediate: true },
);

// Load sync status
async function loadSyncStatus() {
  try {
    syncStatus.value = await fetchSyncStatus();
  } catch (err) {
    console.error("Failed to load sync status:", err);
  }
}

// Re-sync a specific conference (no git clone needed!)
async function resyncConference(conf, year) {
  try {
    await ElMessageBox.confirm(
      `Re-sync ${conf.toUpperCase()} ${year}? This will update data and recompute rating diffs.`,
      "Confirm Re-sync",
      {
        confirmButtonText: "Re-sync",
        cancelButtonText: "Cancel",
        type: "info",
      },
    );
  } catch {
    return; // User cancelled
  }

  downloading.value = true;
  downloadProgress.value = `Re-syncing ${conf} ${year}...`;

  try {
    const result = await syncConference(conf, year);

    if (result.success) {
      ElMessage.success(`Re-synced ${conf} ${year}`);
      downloadProgress.value = "Done!";

      // Reload store data
      await conferenceStore.loadInitialData();

      // If this is the current conference, refresh
      if (
        conferenceStore.selectedConference === conf &&
        conferenceStore.selectedYear === year
      ) {
        // Trigger reload by re-setting
        conferenceStore.setConference(conf, year);
      }
    } else {
      ElMessage.error(result.error || "Re-sync failed");
    }
  } catch (err) {
    console.error("Re-sync error:", err);
    ElMessage.error("Failed to re-sync conference data");
  } finally {
    downloading.value = false;
    downloadProgress.value = "";
  }
}

// Sync all downloaded conferences (no git clone needed!)
async function syncAllConferences() {
  if (downloadedConferences.value.length === 0) {
    ElMessage.warning("No conferences to sync");
    return;
  }

  try {
    await ElMessageBox.confirm(
      `Re-sync all ${downloadedConferences.value.length} downloaded conferences? This may take a while.`,
      "Confirm Sync All",
      {
        confirmButtonText: "Sync All",
        cancelButtonText: "Cancel",
        type: "warning",
      },
    );
  } catch {
    return; // User cancelled
  }

  downloading.value = true;

  try {
    for (let i = 0; i < downloadedConferences.value.length; i++) {
      const { conference, year } = downloadedConferences.value[i];
      downloadProgress.value = `Syncing ${conference} ${year} (${i + 1}/${downloadedConferences.value.length})...`;

      try {
        await syncConference(conference, year);
      } catch (err) {
        console.error(`Failed to sync ${conference} ${year}:`, err);
      }
    }

    ElMessage.success("All conferences synced!");
    downloadProgress.value = "Done!";

    // Reload store data
    await conferenceStore.loadInitialData();
  } catch (err) {
    console.error("Sync all error:", err);
    ElMessage.error("Failed to sync conferences");
  } finally {
    downloading.value = false;
    setTimeout(() => {
      downloadProgress.value = "";
    }, 2000);
  }
}

// Load conferences on mount
onMounted(() => {
  loadAvailableConferences();
  loadSyncStatus();
});
</script>

<template>
  <div class="home-page">
    <section class="hero">
      <h1 class="hero-title">KohakuPaper</h1>
      <p class="hero-subtitle">
        Local Paper Copilot for Academic Conference Analysis
      </p>
    </section>

    <!-- Conference Selector Section -->
    <section class="conference-selector">
      <h2>Select Conference</h2>
      <div class="selector-row" v-loading="conferencesLoading">
        <el-select
          v-model="selectedDownloadConf"
          placeholder="Conference"
          filterable
          class="conf-select"
          @change="selectedDownloadYear = null"
        >
          <el-option
            v-for="conf in availableConferences"
            :key="conf.conference"
            :label="conf.conference.toUpperCase()"
            :value="conf.conference"
          />
        </el-select>

        <el-select
          v-model="selectedDownloadYear"
          placeholder="Year"
          :disabled="!selectedDownloadConf"
          class="year-select"
        >
          <el-option
            v-for="yearInfo in availableYears"
            :key="yearInfo.year"
            :label="`${yearInfo.year}${yearInfo.downloaded ? ' (local)' : ''}`"
            :value="yearInfo.year"
          >
            <span>{{ yearInfo.year }}</span>
            <span v-if="yearInfo.downloaded" class="local-badge">local</span>
            <span v-else class="size-badge">{{ yearInfo.size_mb }} MB</span>
          </el-option>
        </el-select>

        <el-button
          v-if="isDownloaded"
          type="primary"
          @click="viewConference"
          :disabled="!selectedDownloadConf || !selectedDownloadYear"
        >
          <span class="i-carbon-view mr-1"></span>
          View
        </el-button>

        <el-button
          v-if="isDownloaded"
          type="warning"
          @click="resyncConference(selectedDownloadConf, selectedDownloadYear)"
          :loading="downloading"
          :disabled="!selectedDownloadConf || !selectedDownloadYear"
        >
          <span class="i-carbon-sync mr-1" v-if="!downloading"></span>
          Re-sync
        </el-button>

        <el-button
          v-if="!isDownloaded"
          type="success"
          @click="downloadConference"
          :loading="downloading"
          :disabled="!selectedDownloadConf || !selectedDownloadYear"
        >
          <span class="i-carbon-download mr-1" v-if="!downloading"></span>
          {{ downloading ? "Downloading..." : "Download" }}
        </el-button>
      </div>

      <div class="download-progress" v-if="downloadProgress">
        {{ downloadProgress }}
      </div>
    </section>

    <!-- Sync Management Section -->
    <section class="sync-management" v-if="downloadedConferences.length > 0">
      <div class="sync-header">
        <h2>Sync Management</h2>
        <div class="sync-actions">
          <el-button
            size="small"
            type="primary"
            @click="syncAllConferences"
            :loading="downloading"
          >
            <span class="i-carbon-sync mr-1" v-if="!downloading"></span>
            Sync All
          </el-button>
        </div>
      </div>

      <div class="sync-status" v-if="syncStatus">
        <span class="status-item">
          <span class="i-carbon-folder mr-1"></span>
          {{ syncStatus.local_count }} local files
        </span>
      </div>

      <div class="downloaded-list">
        <div
          v-for="item in downloadedConferences"
          :key="`${item.conference}-${item.year}`"
          class="downloaded-item"
        >
          <span class="item-name"
            >{{ item.conference.toUpperCase() }} {{ item.year }}</span
          >
          <span class="item-size">{{ item.size_mb }} MB</span>
          <el-button
            size="small"
            text
            type="primary"
            @click="resyncConference(item.conference, item.year)"
            :disabled="downloading"
          >
            <span class="i-carbon-sync"></span>
          </el-button>
        </div>
      </div>
    </section>

    <!-- Current Conference Stats -->
    <section class="content" v-if="conferenceStore.hasLocalData">
      <div class="current-conf-header">
        <h3>
          {{ conferenceStore.selectedConferenceName }}
          <span v-if="conferenceStore.selectedYear"
            >({{ conferenceStore.selectedYear }})</span
          >
        </h3>
      </div>

      <div class="stats-grid" v-if="summary">
        <div class="stat-card">
          <div class="stat-value">
            {{ summary.total_papers?.toLocaleString() || 0 }}
          </div>
          <div class="stat-label">Total Papers</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">
            {{ summary.avg_rating?.toFixed(2) || "N/A" }}
          </div>
          <div class="stat-label">Average Rating</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">
            {{ summary.accepted_count?.toLocaleString() || 0 }}
          </div>
          <div class="stat-label">Accepted</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">
            {{ summary.rejected_count?.toLocaleString() || 0 }}
          </div>
          <div class="stat-label">Rejected/Withdrawn</div>
        </div>
      </div>

      <div class="actions">
        <el-button type="primary" size="large" @click="goToConference">
          <span class="i-carbon-chart-histogram mr-2"></span>
          View Detailed Analysis
        </el-button>
      </div>
    </section>

    <!-- No Data Hint -->
    <section
      class="no-data"
      v-if="!conferenceStore.hasLocalData && !conferencesLoading"
    >
      <div class="no-data-icon i-carbon-cloud-download"></div>
      <p>Select a conference above and download to get started</p>
    </section>
  </div>
</template>

<style scoped>
.home-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}

.hero {
  text-align: center;
  padding: 2rem 0 1rem;
}

.hero-title {
  font-size: 2.5rem;
  font-weight: 700;
  color: var(--el-text-color-primary);
  margin-bottom: 0.5rem;
}

.hero-subtitle {
  font-size: 1.125rem;
  color: var(--el-text-color-secondary);
}

/* Conference Selector */
.conference-selector {
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-light);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 2rem;
}

.conference-selector h2 {
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: var(--el-text-color-primary);
}

.selector-row {
  display: flex;
  gap: 1rem;
  align-items: center;
  flex-wrap: wrap;
}

.conf-select {
  width: 180px;
}

.year-select {
  width: 160px;
}

.local-badge {
  margin-left: auto;
  font-size: 0.75rem;
  color: var(--el-color-success);
  background: var(--el-color-success-light-9);
  padding: 2px 6px;
  border-radius: 4px;
}

.size-badge {
  margin-left: auto;
  font-size: 0.75rem;
  color: var(--el-text-color-secondary);
}

.download-progress {
  margin-top: 0.75rem;
  font-size: 0.875rem;
  color: var(--el-color-primary);
}

/* Sync Management */
.sync-management {
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-light);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 2rem;
}

.sync-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.sync-header h2 {
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--el-text-color-primary);
  margin: 0;
}

.sync-actions {
  display: flex;
  gap: 0.5rem;
}

.sync-status {
  display: flex;
  gap: 1.5rem;
  margin-bottom: 1rem;
  padding: 0.75rem;
  background: var(--el-fill-color-light);
  border-radius: 8px;
}

.status-item {
  display: flex;
  align-items: center;
  font-size: 0.875rem;
  color: var(--el-text-color-secondary);
}

.status-item.status-ok {
  color: var(--el-color-success);
}

.status-item.status-warn {
  color: var(--el-color-warning);
}

.status-item.status-error {
  color: var(--el-color-danger);
}

.downloaded-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.downloaded-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  background: var(--el-fill-color-light);
  border-radius: 6px;
  font-size: 0.875rem;
}

.downloaded-item .item-name {
  font-weight: 500;
  color: var(--el-text-color-primary);
}

.downloaded-item .item-size {
  color: var(--el-text-color-secondary);
  font-size: 0.75rem;
}

/* Current Conference Header */
.current-conf-header {
  margin-bottom: 1.5rem;
}

.current-conf-header h3 {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--el-text-color-primary);
}

.content {
  margin-top: 1rem;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.stat-card {
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-light);
  border-radius: 12px;
  padding: 1.25rem;
  text-align: center;
  transition: box-shadow 0.2s;
}

.stat-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.stat-value {
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--el-color-primary);
}

.stat-label {
  font-size: 0.8rem;
  color: var(--el-text-color-secondary);
  margin-top: 0.5rem;
}

.actions {
  text-align: center;
  margin: 1.5rem 0;
}

.no-data {
  text-align: center;
  padding: 3rem 2rem;
}

.no-data-icon {
  font-size: 3rem;
  color: var(--el-text-color-placeholder);
  margin-bottom: 1rem;
}

.no-data p {
  color: var(--el-text-color-secondary);
}
</style>
