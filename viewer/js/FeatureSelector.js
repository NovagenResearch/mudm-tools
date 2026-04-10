/**
 * Feature selector sidebar — searchable, filterable checkbox list.
 *
 * Loads features.json and renders a filterable list. Fires a callback
 * whenever the selection changes with the set of selected feature names.
 */

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
        this.checkboxes = new Map();  // name → checkbox element
        this._allItems = [];         // all list item elements
        this._searchQuery = '';
        // Filter state
        this._filterAttr = '';       // current filter attribute key
        this._filterType = '';       // 'categorical' or 'numeric'
        this._filterValues = new Set(); // checked categorical values (empty = no filter)
        this._filterMin = null;      // numeric min (null = unbounded)
        this._filterMax = null;      // numeric max (null = unbounded)
        this._idFields = new Set();  // fields to exclude (from features.json id_fields)
    }

    async init(featuresUrl, idFields = []) {
        const resp = await fetch(featuresUrl);
        const data = await resp.json();
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
        this.selected.clear();
        this.checkboxes.clear();
        this._allItems = [];
        this._searchQuery = '';
        this._filterAttr = '';
        this._filterType = '';
        this._filterValues.clear();
        this._filterMin = null;
        this._filterMax = null;
        this._render();
    }

    _render() {
        this.container.innerHTML = '';

        // Search input
        const search = document.createElement('input');
        search.type = 'text';
        search.placeholder = 'Search features...';
        search.className = 'feature-search';
        search.addEventListener('input', () => {
            this._searchQuery = search.value;
            this._applyFilters();
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
        count.textContent = `0 / ${Object.keys(this.features).length}`;
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

        // List
        const list = document.createElement('div');
        list.className = 'feature-list';

        const names = Object.keys(this.features).sort();
        for (const name of names) {
            const feat = this.features[name];
            const item = document.createElement('label');
            item.className = 'feature-item';
            item.dataset.name = name.toLowerCase();

            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.addEventListener('change', () => this._toggle(name, cb.checked));
            this.checkboxes.set(name, cb);

            const swatch = document.createElement('span');
            swatch.className = 'feature-swatch';
            swatch.style.backgroundColor = feat.color || '#888';

            const label = document.createElement('span');
            label.className = 'feature-name';
            label.textContent = name;
            const tileCount = typeof feat.tiles === 'object'
                ? Object.values(feat.tiles).reduce((s, a) => s + a.length, 0) : 0;
            label.title = `${feat.acronym || ''} (${tileCount} tiles)`;

            item.appendChild(cb);
            item.appendChild(swatch);
            item.appendChild(label);
            list.appendChild(item);
            this._allItems.push(item);
        }

        this.container.appendChild(list);
    }

    // --- Filter logic ---

    /**
     * Discover filterable attributes. Classifies as categorical or numeric.
     * Excludes ID-like attributes (unique values > 50% of features).
     */
    _discoverFilterAttrs() {
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

        select.addEventListener('change', () => {
            this._filterAttr = select.value;
            this._filterValues.clear();
            this._filterMin = null;
            this._filterMax = null;
            const attr = attrs.find(a => a.key === select.value);
            this._filterType = attr?.type || '';
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
        minLabel.textContent = '\u2265'; // ≥
        minLabel.className = 'filter-numeric-label';
        const minInput = makeInput(String(min));

        const maxLabel = document.createElement('span');
        maxLabel.textContent = '\u2264'; // ≤
        maxLabel.className = 'filter-numeric-label';
        const maxInput = makeInput(String(max));

        const update = () => {
            this._filterMin = minInput.value !== '' ? parseFloat(minInput.value) : null;
            this._filterMax = maxInput.value !== '' ? parseFloat(maxInput.value) : null;
            this._applyFilters();
        };
        minInput.addEventListener('input', update);
        maxInput.addEventListener('input', update);

        row.appendChild(minLabel);
        row.appendChild(minInput);
        row.appendChild(maxLabel);
        row.appendChild(maxInput);

        const rangeHint = document.createElement('div');
        rangeHint.className = 'filter-range-hint';
        rangeHint.textContent = `Range: ${min.toLocaleString()} \u2013 ${max.toLocaleString()}`;

        container.appendChild(row);
        container.appendChild(rangeHint);
    }

    /**
     * Apply search + attribute filter to feature list visibility.
     */
    _applyFilters() {
        const q = this._searchQuery.toLowerCase().trim();
        for (const item of this._allItems) {
            const name = item.querySelector('.feature-name')?.textContent || '';
            // Search filter
            const matchesSearch = !q || name.toLowerCase().includes(q);
            // Attribute filter
            const matchesFilter = this._matchesFilter(name);
            item.style.display = (matchesSearch && matchesFilter) ? '' : 'none';
        }
    }

    _matchesFilter(name) {
        if (!this._filterAttr) return true; // no filter active

        const feat = this.features[name];
        if (!feat) return false;
        const val = feat[this._filterAttr];

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

    _selectAllVisible() {
        for (const item of this._allItems) {
            if (item.style.display !== 'none') {
                const name = item.querySelector('.feature-name').textContent;
                const cb = this.checkboxes.get(name);
                if (cb && !cb.checked) {
                    cb.checked = true;
                    this.selected.add(name);
                }
            }
        }
        this._updateCount();
        this.onSelectionChange(this.selected);
    }

    _clearAll() {
        for (const [name, cb] of this.checkboxes) {
            cb.checked = false;
            this.selected.delete(name);
        }
        this._updateCount();
        this.onSelectionChange(this.selected);
    }

    _updateCount() {
        const el = document.getElementById('feature-count');
        if (el) el.textContent = `${this.selected.size} / ${Object.keys(this.features).length}`;
    }

    /**
     * Update sidebar swatch colors. Pass null to restore original colors.
     * @param {Map<string, string>|null} nameColorMap - feature name → hex color
     */
    updateSwatchColors(nameColorMap) {
        for (const item of this._allItems) {
            const swatch = item.querySelector('.feature-swatch');
            if (!swatch) continue;
            const name = item.querySelector('.feature-name')?.textContent;
            if (!name) continue;
            if (nameColorMap && nameColorMap.has(name)) {
                swatch.style.backgroundColor = nameColorMap.get(name);
            } else {
                const feat = this.features[name];
                swatch.style.backgroundColor = feat?.color || '#888';
            }
        }
    }
}
