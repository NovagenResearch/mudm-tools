/**
 * Feature selector sidebar — searchable, filterable, VIRTUALIZED checkbox list.
 *
 * Loads features.json and renders a filterable list. Fires a callback
 * whenever the selection changes with the set of selected feature names.
 *
 * Scales to 100k+ features: the list is windowed — only the rows in the
 * scroll viewport (plus a small overscan) exist in the DOM at any time, so
 * DOM node count is O(viewport), not O(features). Search/filter run over an
 * in-memory model and re-render only the window. A single delegated change
 * listener on the list container handles all checkboxes (no per-row listeners).
 */

const ROW_H = 26;          // px; fixed row height (enforced inline) so windowing math is exact
const OVERSCAN = 6;        // extra rows rendered above/below the viewport to hide scroll seams
const SEARCH_DEBOUNCE_MS = 150;

export class FeatureSelector {
    /**
     * @param {HTMLElement} container - DOM element to render into
     * @param {function(Set<string>): void} onSelectionChange - callback with selected names
     */
    constructor(container, onSelectionChange) {
        this.container = container;
        this.onSelectionChange = onSelectionChange;
        this.features = {};          // name → {color, acronym, ccf_id, tiles, ...}
        this.selected = new Set();

        // Virtualized-list model
        this._model = [];            // [{name, color, lower, props}] sorted by name
        this._filtered = [];         // subset of _model passing search + filter (entries)
        this._listEl = null;         // scroll viewport (.feature-list)
        this._sizer = null;          // tall inner spacer; rows are absolutely positioned inside
        this._scrollRaf = 0;         // rAF handle coalescing scroll-driven re-renders
        this._searchTimer = 0;       // debounce handle for search input
        this._filterTimer = 0;       // debounce handle for numeric filter inputs
        this._swatchColors = null;   // optional name→color override (color-by mode)

        this._searchQuery = '';
        // Filter state
        this._filterAttr = '';       // current filter attribute key
        this._filterType = '';       // 'categorical' or 'numeric'
        this._filterValues = new Set(); // checked categorical values (empty = no filter)
        this._filterMin = null;      // numeric min (null = unbounded)
        this._filterMax = null;      // numeric max (null = unbounded)
        this._idFields = new Set();  // fields to exclude (from features.json id_fields)

        // Facet-store path (schema + columnar parquet from the muDM TileModel descriptor) — set by
        // init() when both are available; null falls back to the features.json-derived path below.
        this._schema = null;         // vector_layers[0] filter-field schema, or null
        this._facetStore = null;     // {baseUrl, href, key}, or null
        this._colCache = {};         // attr key → Map(name -> value), lazily populated by _loadColumn
    }

    /**
     * @param {string|object} featuresSource - a URL to fetch features.json from, an already-parsed
     *   features document, OR (in the schema+facetStore path) unused — the feature list is read
     *   column-wise from the facet-store parquet instead.
     * @param {string[]} idFields - id-like fields to exclude from the filter panel (features.json path).
     * @param {object|null} schema - vector_layers[0] filter-field schema ({fields, fieldenums,
     *   fieldranges}) from the muDM TileModel descriptor, or null to use the features.json path.
     * @param {{baseUrl: string, href: string, key: string}|null} facetStore - the role="facets"
     *   columnar parquet asset, or null to use the features.json path.
     */
    async init(featuresSource, idFields = [], schema = null, facetStore = null) {
        this._schema = schema;
        this._facetStore = facetStore;
        this._colCache = {};
        this.selected.clear();
        this._swatchColors = null;
        this._searchQuery = '';
        this._filterAttr = '';
        this._filterType = '';
        this._filterValues.clear();
        this._filterMin = null;
        this._filterMax = null;

        if (schema && facetStore) {
            // Facet-store path: feature list + filter values are read by-column from the parquet
            // (hyparquet), not parsed out of the monolithic features.json.
            this.features = {};                       // not used in the facet path
            this._idFields = new Set(idFields);
            const { asyncBufferFromUrl, parquetMetadataAsync, parquetReadObjects } =
                await import('./vendor/hyparquet.min.js');
            this._pqRead = parquetReadObjects;
            this._facetKey = facetStore.key;
            this._pqFile = await asyncBufferFromUrl({ url: facetStore.baseUrl + facetStore.href });
            this._pqMeta = await parquetMetadataAsync(this._pqFile);
            const rows = await parquetReadObjects({ file: this._pqFile, metadata: this._pqMeta,
                                                    columns: [facetStore.key, 'color'] });
            this._model = rows.map(r => {
                const name = String(r[facetStore.key]);
                return { name, color: r.color || '#888', lower: name.toLowerCase(), props: null };
            }).sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
            this._filtered = this._model;
            this._render();
            return;
        }

        // Fallback: parse the (already-fetched, or fetched-here) features.json document.
        let data;
        if (typeof featuresSource === 'string') {
            const resp = await fetch(featuresSource);
            data = await resp.json();
        } else {
            data = featuresSource;
        }
        // Use idFields parameter if provided, fall back to features.json metadata
        if (idFields.length > 0) {
            this._idFields = new Set(idFields);
        } else {
            const collProps = data.properties ?? {};
            this._idFields = new Set(collProps.id_fields ?? data.id_fields ?? []);
        }
        if (Array.isArray(data.features)) {
            // MicroJSON format: array of {type, id, geometry, properties}
            this.features = {};
            for (const feat of data.features) {
                const name = feat.id ?? feat.properties?.name ?? '';
                if (name) this.features[name] = feat.properties;
            }
        } else {
            // Legacy format: dict keyed by name
            this.features = data.features;
        }
        this._buildModel();
        this._render();
    }

    /** Build the sorted in-memory model once; rendering is derived from it. */
    _buildModel() {
        const names = Object.keys(this.features).sort();
        this._model = names.map(name => {
            const feat = this.features[name];
            return { name, color: feat?.color || '#888', lower: name.toLowerCase(), props: feat };
        });
        this._filtered = this._model;
    }

    _render() {
        this.container.innerHTML = '';

        // Search input (debounced — drives filtering off the in-memory model)
        const search = document.createElement('input');
        search.type = 'text';
        search.placeholder = 'Search features...';
        search.className = 'feature-search';
        search.addEventListener('input', () => {
            clearTimeout(this._searchTimer);
            this._searchTimer = setTimeout(() => {
                this._searchQuery = search.value;
                this._applyFilters();
            }, SEARCH_DEBOUNCE_MS);
        });
        this.container.appendChild(search);

        // Filter row
        this._renderFilterRow();

        // Toolbar
        const toolbar = document.createElement('div');
        toolbar.className = 'feature-toolbar';

        const count = document.createElement('span');
        count.className = 'feature-count';
        count.id = 'feature-count';
        count.textContent = `0 / ${this._model.length}`;
        toolbar.appendChild(count);

        const selectAllBtn = document.createElement('button');
        selectAllBtn.textContent = 'All';
        selectAllBtn.title = 'Select all visible';
        selectAllBtn.addEventListener('click', () => this._selectAllVisible());
        toolbar.appendChild(selectAllBtn);

        const clearBtn = document.createElement('button');
        clearBtn.textContent = 'Clear';
        clearBtn.title = 'Clear selection';
        clearBtn.addEventListener('click', () => this._clearAll());
        toolbar.appendChild(clearBtn);

        this.container.appendChild(toolbar);

        // Virtualized list: a scrolling viewport containing a tall sizer; only the
        // rows in view (plus overscan) are materialized as absolutely-positioned
        // children of the sizer.
        const list = document.createElement('div');
        list.className = 'feature-list';
        const sizer = document.createElement('div');
        sizer.className = 'feature-list-sizer';
        sizer.style.position = 'relative';
        sizer.style.width = '100%';
        list.appendChild(sizer);
        this._listEl = list;
        this._sizer = sizer;

        // One delegated change listener for ALL checkboxes (rows are recycled).
        list.addEventListener('change', (e) => {
            const cb = e.target;
            if (!cb || cb.type !== 'checkbox') return;
            const item = cb.closest('.feature-item');
            if (item) this._toggle(item.dataset.name, cb.checked);
        });
        // Re-window on scroll (coalesced to one render per animation frame).
        list.addEventListener('scroll', () => {
            if (this._scrollRaf) return;
            this._scrollRaf = requestAnimationFrame(() => {
                this._scrollRaf = 0;
                this._renderWindow();
            });
        });
        // Re-window when the sidebar resizes (viewport row count changes).
        if (typeof ResizeObserver !== 'undefined') {
            this._resizeObserver?.disconnect();
            this._resizeObserver = new ResizeObserver(() => this._renderWindow());
            this._resizeObserver.observe(list);
        }

        this.container.appendChild(list);

        this._applyFilters();
    }

    /** Build one row element for a model entry, positioned at the given top offset. */
    _makeRow(entry, topPx) {
        const item = document.createElement('label');
        item.className = 'feature-item';
        item.dataset.name = entry.name;
        item.style.position = 'absolute';
        item.style.top = topPx + 'px';
        item.style.left = '0';
        item.style.right = '0';
        item.style.height = ROW_H + 'px';
        item.style.boxSizing = 'border-box';

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = this.selected.has(entry.name);

        const swatch = document.createElement('span');
        swatch.className = 'feature-swatch';
        swatch.style.backgroundColor = this._swatchColor(entry);

        const label = document.createElement('span');
        label.className = 'feature-name';
        label.textContent = entry.name;
        const tiles = entry.props?.tiles;
        const tileCount = typeof tiles === 'object' && tiles
            ? Object.values(tiles).reduce((s, a) => s + a.length, 0) : 0;
        label.title = `${entry.props?.acronym || ''} (${tileCount} tiles)`;

        item.appendChild(cb);
        item.appendChild(swatch);
        item.appendChild(label);
        return item;
    }

    /** Effective swatch color for an entry (honors color-by override if set). */
    _swatchColor(entry) {
        if (this._swatchColors && this._swatchColors.has(entry.name)) {
            return this._swatchColors.get(entry.name);
        }
        return entry.color;
    }

    /** Render only the rows currently in the scroll viewport (+ overscan). */
    _renderWindow() {
        const list = this._listEl;
        if (!list || !this._sizer) return;
        const total = this._filtered.length;
        const viewH = list.clientHeight || 400;
        const scrollTop = list.scrollTop;

        let start = Math.floor(scrollTop / ROW_H) - OVERSCAN;
        if (start < 0) start = 0;
        const visible = Math.ceil(viewH / ROW_H) + OVERSCAN * 2;
        const end = Math.min(total, start + visible);

        const frag = document.createDocumentFragment();
        for (let i = start; i < end; i++) {
            frag.appendChild(this._makeRow(this._filtered[i], i * ROW_H));
        }
        this._sizer.replaceChildren(frag);
    }

    // --- Filter logic ---

    /**
     * Discover filterable attributes. Classifies as categorical or numeric.
     * Excludes ID-like attributes (unique values > 50% of features).
     */
    _discoverFilterAttrs() {
        if (this._schema) {
            // Schema-driven path: the descriptor already declares field types/enums/ranges, so no
            // per-feature scan is needed (and none of the feature properties are even loaded).
            const fields = this._schema.fields || {}, enums = this._schema.fieldenums || {}, ranges = this._schema.fieldranges || {};
            const out = [];
            for (const [key, type] of Object.entries(fields)) {
                const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                if (type === 'number' && ranges[key]) out.push({ key, label, type: 'numeric', values: ranges[key] });   // [min,max]
                else if (enums[key]) out.push({ key, label, type: 'categorical', values: enums[key] });
            }
            return out.sort((a, b) => a.key.localeCompare(b.key));
        }
        const SKIP = new Set(['color', 'tiles', 'acronym']);
        for (const f of this._idFields) SKIP.add(f);
        const attrMeta = {}; // key → {values: Set, allNumeric: bool}

        for (const feat of Object.values(this.features)) {
            for (const [key, val] of Object.entries(feat)) {
                if (SKIP.has(key)) continue;
                if (val === null || val === undefined || typeof val === 'object') continue;
                if (!attrMeta[key]) attrMeta[key] = { values: new Set(), allNumeric: true };
                // Try to parse as number (handles string-encoded numerics)
                const num = typeof val === 'number' ? val : Number(val);
                if (isNaN(num)) {
                    attrMeta[key].allNumeric = false;
                    attrMeta[key].values.add(val);
                } else {
                    attrMeta[key].values.add(num);
                }
            }
        }

        const result = [];
        for (const [key, meta] of Object.entries(attrMeta)) {
            if (meta.values.size < 2) continue;
            const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            // Classify: numeric if all values are numbers
            const type = meta.allNumeric ? 'numeric' : 'categorical';
            const values = [...meta.values].map(v => type === 'numeric' ? v : String(v));
            if (type === 'numeric') {
                values.sort((a, b) => a - b);
            } else {
                values.sort();
            }
            result.push({ key, label, type, values });
        }
        return result.sort((a, b) => a.key.localeCompare(b.key));
    }

    /**
     * Lazily fetch one column from the facet-store parquet (keyed by the facet key column) and cache
     * it as a name -> value Map. Called before a facet-path filter is applied, since _applyFilters'
     * scan over _matchesFilter is synchronous and needs the column already resident.
     */
    async _loadColumn(attr) {
        if (this._colCache[attr]) return;
        const rows = await this._pqRead({ file: this._pqFile, metadata: this._pqMeta,
                                          columns: [this._facetKey, attr] });
        const m = new Map();
        for (const r of rows) m.set(String(r[this._facetKey]), r[attr]);
        this._colCache[attr] = m;
    }

    _renderFilterRow() {
        const row = document.createElement('div');
        row.className = 'feature-filter-row';

        const attrs = this._discoverFilterAttrs();
        if (attrs.length === 0) return;
        this._filterAttrs = attrs;

        const select = document.createElement('select');
        select.className = 'feature-filter-select';
        const none = document.createElement('option');
        none.value = '';
        none.textContent = 'Filter by...';
        select.appendChild(none);
        for (const attr of attrs) {
            const opt = document.createElement('option');
            opt.value = attr.key;
            const suffix = attr.type === 'numeric' ? ' (range)' : ` (${attr.values.length})`;
            opt.textContent = attr.label + suffix;
            select.appendChild(opt);
        }

        const valContainer = document.createElement('div');
        valContainer.className = 'feature-filter-values';

        select.addEventListener('change', async () => {
            this._filterAttr = select.value;
            this._filterValues.clear(); this._filterMin = null; this._filterMax = null;
            const attr = attrs.find(a => a.key === select.value);
            this._filterType = attr?.type || '';
            // Facet path: the filter loop (_applyFilters -> _matchesFilter) is synchronous, so the
            // active column must already be in _colCache before we (re)apply the filter.
            if (this._facetStore && this._filterAttr) await this._loadColumn(this._filterAttr);
            this._renderFilterControls(valContainer);
            this._applyFilters();
        });

        row.appendChild(select);
        row.appendChild(valContainer);
        this.container.appendChild(row);
    }

    _renderFilterControls(container) {
        container.innerHTML = '';
        if (!this._filterAttr) return;

        const attr = this._filterAttrs.find(a => a.key === this._filterAttr);
        if (!attr) return;

        if (attr.type === 'numeric') {
            this._renderNumericFilter(container, attr);
        } else {
            this._renderCategoricalFilter(container, attr);
        }
    }

    _renderCategoricalFilter(container, attr) {
        // Search input for filtering values
        const valSearch = document.createElement('input');
        valSearch.type = 'text';
        valSearch.placeholder = `Search ${attr.label.toLowerCase()}...`;
        valSearch.className = 'filter-value-search';
        container.appendChild(valSearch);

        // All / Clear toggle buttons
        const btnRow = document.createElement('div');
        btnRow.className = 'filter-btn-row';
        const allBtn = document.createElement('button');
        allBtn.textContent = 'All';
        allBtn.className = 'filter-toggle-btn';
        const clearBtn = document.createElement('button');
        clearBtn.textContent = 'Clear';
        clearBtn.className = 'filter-toggle-btn';
        btnRow.appendChild(allBtn);
        btnRow.appendChild(clearBtn);
        container.appendChild(btnRow);

        const valList = document.createElement('div');
        valList.className = 'filter-value-list';

        const checkboxes = []; // {cb, val, lbl}
        for (const val of attr.values) {
            const lbl = document.createElement('label');
            lbl.className = 'filter-value-item';
            lbl.dataset.val = String(val).toLowerCase();
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.addEventListener('change', () => {
                if (cb.checked) {
                    this._filterValues.add(String(val));
                } else {
                    this._filterValues.delete(String(val));
                }
                this._applyFilters();
            });
            const span = document.createElement('span');
            span.textContent = val;
            span.title = val;
            lbl.appendChild(cb);
            lbl.appendChild(span);
            valList.appendChild(lbl);
            checkboxes.push({ cb, val: String(val), lbl });
        }

        allBtn.addEventListener('click', () => {
            for (const { cb, val, lbl } of checkboxes) {
                if (lbl.style.display === 'none') continue; // skip search-hidden
                cb.checked = true;
                this._filterValues.add(val);
            }
            this._applyFilters();
        });

        clearBtn.addEventListener('click', () => {
            for (const { cb, val } of checkboxes) {
                cb.checked = false;
                this._filterValues.delete(val);
            }
            this._applyFilters();
        });

        valSearch.addEventListener('input', () => {
            const q = valSearch.value.toLowerCase().trim();
            for (const { lbl } of checkboxes) {
                lbl.style.display = (!q || lbl.dataset.val.includes(q)) ? '' : 'none';
            }
        });

        container.appendChild(valList);
    }

    _renderNumericFilter(container, attr) {
        const min = attr.values[0];
        const max = attr.values[attr.values.length - 1];

        const row = document.createElement('div');
        row.className = 'filter-numeric-row';

        const makeInput = (placeholder, defaultVal) => {
            const inp = document.createElement('input');
            inp.type = 'number';
            inp.className = 'filter-numeric-input';
            inp.placeholder = placeholder;
            inp.step = 'any';
            return inp;
        };

        const minLabel = document.createElement('span');
        minLabel.textContent = '≥'; // ≥
        minLabel.className = 'filter-numeric-label';
        const minInput = makeInput(String(min));

        const maxLabel = document.createElement('span');
        maxLabel.textContent = '≤'; // ≤
        maxLabel.className = 'filter-numeric-label';
        const maxInput = makeInput(String(max));

        const update = () => {
            // Debounced: numeric inputs fire on every keystroke; each _applyFilters
            // is an O(features) scan, so coalesce rapid typing.
            clearTimeout(this._filterTimer);
            this._filterTimer = setTimeout(() => {
                this._filterMin = minInput.value !== '' ? parseFloat(minInput.value) : null;
                this._filterMax = maxInput.value !== '' ? parseFloat(maxInput.value) : null;
                this._applyFilters();
            }, SEARCH_DEBOUNCE_MS);
        };
        minInput.addEventListener('input', update);
        maxInput.addEventListener('input', update);

        row.appendChild(minLabel);
        row.appendChild(minInput);
        row.appendChild(maxLabel);
        row.appendChild(maxInput);

        const rangeHint = document.createElement('div');
        rangeHint.className = 'filter-range-hint';
        rangeHint.textContent = `Range: ${min.toLocaleString()} – ${max.toLocaleString()}`;

        container.appendChild(row);
        container.appendChild(rangeHint);
    }

    /**
     * Recompute the filtered set (search + attribute filter) from the model and
     * re-render the visible window. O(features) over strings — no DOM touched
     * except the ~viewport rows produced by _renderWindow().
     */
    _applyFilters() {
        const q = this._searchQuery.toLowerCase().trim();
        const hasFilter = !!this._filterAttr;
        if (!q && !hasFilter) {
            this._filtered = this._model;
        } else {
            const out = [];
            for (const entry of this._model) {
                if (q && !entry.lower.includes(q)) continue;
                if (hasFilter && !this._matchesFilter(entry.name)) continue;
                out.push(entry);
            }
            this._filtered = out;
        }
        if (this._sizer) this._sizer.style.height = (this._filtered.length * ROW_H) + 'px';
        if (this._listEl) this._listEl.scrollTop = 0;
        this._renderWindow();
    }

    _matchesFilter(name) {
        if (!this._filterAttr) return true; // no filter active

        let val;
        if (this._facetStore) {
            val = this._colCache[this._filterAttr]?.get(name);
        } else {
            const feat = this.features[name];
            if (!feat) return false;
            val = feat[this._filterAttr];
        }

        if (this._filterType === 'numeric') {
            if (this._filterMin === null && this._filterMax === null) return true;
            if (val === null || val === undefined) return false;
            const num = Number(val);
            if (isNaN(num)) return false;
            if (this._filterMin !== null && num < this._filterMin) return false;
            if (this._filterMax !== null && num > this._filterMax) return false;
            return true;
        } else {
            // Categorical: empty set = no filter (show all)
            if (this._filterValues.size === 0) return true;
            return this._filterValues.has(String(val ?? ''));
        }
    }

    // --- Selection logic ---

    _toggle(name, checked) {
        if (checked) {
            this.selected.add(name);
        } else {
            this.selected.delete(name);
        }
        this._updateCount();
        this.onSelectionChange(this.selected);
    }

    /** Select every feature currently passing search + filter (the "visible" set). */
    _selectAllVisible() {
        for (const entry of this._filtered) {
            this.selected.add(entry.name);
        }
        this._updateCount();
        this._renderWindow();   // refresh checkbox state of on-screen rows
        this.onSelectionChange(this.selected);
    }

    _clearAll() {
        this.selected.clear();
        this._updateCount();
        this._renderWindow();
        this.onSelectionChange(this.selected);
    }

    _updateCount() {
        const el = document.getElementById('feature-count');
        if (el) el.textContent = `${this.selected.size} / ${this._model.length}`;
    }

    /**
     * Update sidebar swatch colors. Pass null to restore original colors.
     * @param {Map<string, string>|null} nameColorMap - feature name → hex color
     */
    updateSwatchColors(nameColorMap) {
        this._swatchColors = nameColorMap || null;
        this._renderWindow();   // only the visible rows need recoloring
    }
}
