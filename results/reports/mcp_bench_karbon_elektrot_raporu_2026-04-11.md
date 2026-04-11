# Azot Katkili Goznekli Karbon Elektrot Deneyi - MCP Yerel Yurutum Raporu

- Tarih: 11 Nisan 2026
- Calisma ortami: Yerel makine (sunucu degil)
- Calistirma araligi (UTC): 2026-04-11T02:50:14Z -> 2026-04-11T02:52:53Z
- GPU dogrulamasi: `resolved_device=/GPU:0` (TensorFlow Metal, Apple GPU)

## 1) Lab Lideri Icin Kisa Yonetici Ozeti
Bu calisma yerelde MCP akisi ile uc asamada tamamlandi: (i) runtime ve veri/alignment kontrolu, (ii) modelin GPU uzerinde yeniden egitimi, (iii) 6-adim tahmin + isletim optimizasyonu + artifact yazimi.

Ana sonuc: sistem calisir durumda ve GPU kullanildi; ancak modelin mevcut hedef degisken seti (repo benchmarki) elektrot-deneyi hedeflerine tam birebir degil. Bu nedenle cikarimlar **proxy-temelli** ve konservatif yorumlandi.

Operasyonel risk sinyali (son 7 gun):
- Tedarik surekliligi: 0.40 (orta)
- Utility guvenilirligi: 0.40 (orta)
- Su-iyon yuku: 0.33 (orta-dusuk)
- Proses kesinti riski: 0.44 (orta)
- Kontaminasyon riski: 0.34 (orta-dusuk)

6 adimlik tahminde proxy sinyaller, gaz-faz donusum performansinda ve verim-retansiyon kanallarinda asagi yonlu kayma, iletkenlik tarafinda hafif bozulma egilimi gosteriyor. Bu nedenle agresif degil, dayanikli bir staged pyrolysis + kontrollu rinse/conditioning takvimi onerildi.

## 2) Kaynak-Temelli Kanit Ozeti (Son 7 Takvim Gunu)
Asagidaki kaynaklar 2026-04-05 ile 2026-04-10 arasindadir ve operasyonel etkiye gore filtrelenmistir.

| Kaynak | Tarih | Kisa Ozet | Operasyonel Iliski | Guven | Supply | Utility | Water-ion | Interruption | Contamination |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| [Newsweek - America Has a Helium Problem](https://www.newsweek.com/america-has-a-helium-problem-11768757) | 2026-04-05 | Helyum/enerji kaynakli arz gerilimi | Inert gaz tampon stok ihtiyaci | 0.64 | 0.72 | 0.35 | 0.15 | 0.58 | 0.20 |
| [EIA Weekly Natural Gas Storage Report](https://ir.eia.gov/ngs/ngs.html?source=content_type%3Areact%7Cfirst_level_url%3Anews%7Csection%3Amain_content%7Cbutton%3Abody_link) | 2026-04-09 | Depolama 1,911 Bcf; ortalamanin ustu | Gaz tabanli utility baskisi goreli dusuk | 0.84 | 0.38 | 0.33 | 0.10 | 0.26 | 0.08 |
| [EIA STEO (Release: 2026-04-07)](https://www.eia.gov/outlooks/steo/report/elec_coal_renew.php) | 2026-04-07 | Kisa vadede enerji/gaz dengesi goreli stabil | Firin penceresi riski orta-dusuk | 0.86 | 0.40 | 0.36 | 0.10 | 0.28 | 0.08 |
| [BLS CPI (2026-04-10)](https://www.bls.gov/news.release/archives/cpi_04102026.htm) | 2026-04-10 | Enerji fiyatlarinda yuksek aylik artis | Maliyet kaynakli operasyon kesinti riski | 0.90 | 0.44 | 0.55 | 0.12 | 0.36 | 0.10 |
| [EPA+FBI+CISA+NSA Su Sistemleri Danismanligi](https://www.epa.gov/newsreleases/epa-fbi-cisa-nsa-issue-joint-cybersecurity-advisory-water-system-regarding-iranian) | 2026-04-07 | Su/atiksu operasyonel bozulma riski | Rinse su kalitesi icin ek kalite kapisi | 0.91 | 0.30 | 0.49 | 0.58 | 0.60 | 0.67 |
| [NOAA Week-2 Hazards Outlook](https://www.cpc.ncep.noaa.gov/products/predictions/threats/threats.php) | 2026-04-10 | Sicaklik/kuraklik/asiri hava riski | Ambient-nem ve utility kesinti etkisi | 0.83 | 0.34 | 0.46 | 0.37 | 0.57 | 0.33 |
| [EPA RCRA Hazardous Waste Exporters](https://www.epa.gov/hwgenerators/information-exporters-resource-conservation-and-recovery-act-rcra-hazardous-waste) | 2026-04-06 | Manifest/uyum disiplin gereksinimi | Asit-rinse atiginda uyumsuzluk kesinti riski | 0.88 | 0.22 | 0.28 | 0.32 | 0.41 | 0.48 |

Belirsizlik notlari:
- Son 7 gunde elektrot-ozel precursor saflik bulteni sinirli.
- Kaynaklar ulusal/bolgesel; kampus-yerel su iletkenligi ve anlik utility loglari yok.
- Gaz/enerji sinyalleri makro seviyeden turetildi; yerel tedarikci SLA dogrulamasi gerekli.

## 3) Sonraki 6 Karar Adimi Icin Tahmin Ozeti

### 3.1 Proxy-Haritalama (Yapisal Not)
Mevcut model hedefleri elektrot deneyine birebir degil. Bu raporda asagidaki esitlestirme yapildi:
- Gaz-faz bozunma davranisi: `h2_co_ratio`, `methane_slip_pct`
- N-retansiyon proxy: `urea_yield_pct`, `ftir_urea_carbonyl_ratio` (ikincil: `ftir_methoxy_ratio`)
- Su/stabilite proxy: `nitrate_mg_l`, `permeate_conductivity_uScm`

Bu, **yapisal bir uzlasidir**; elektrot-ozel online gaz/N-analizi geldiginde model hedefleri dogrudan guncellenmelidir.

### 3.2 6 Adimlik Trend
Step-1 -> Step-6 degisimleri:
- `h2_co_ratio`: 2.4372 -> 2.3864 (dusuyor)
- `methane_slip_pct`: 3.9929 -> 4.3716 (yukseliyor)
- `urea_yield_pct` (N-retansiyon proxy): 87.8385 -> 84.7009 (dusuyor)
- `ftir_urea_carbonyl_ratio`: 1.2950 -> 1.2641 (dusuyor)
- `nitrate_mg_l`: 4.1028 -> 4.0834 (yaklasik stabil)
- `permeate_conductivity_uScm`: 209.19 -> 211.59 (hafif bozulma)

Operasyonel yorum:
- Gaz-faz kanalinda tam donusum zayiflama egilimi var (methane slip artisi).
- N-retansiyon/yield proxy asagi kayiyor; pik sicakligi asiri zorlamamak gerekir.
- Rinse su kalitesi/iletkenlik kapisi, son adimlarda kritik.

## 4) Isletim Takvimi

### 4.1 Onerilen Takvim (Robust, konservatif)
Amaç: N-retansiyon + iletkenlik + verim + wash stabilitesi dengesi.

**Asama A - Hazirlik (T-60 -> T0)**
- Precursor karisimi: N-kaynak + karbon feedstock (homojen karisim, kuru ortam)
- Firin leak-check ve N2 hat testi
- Su kalite kapisi: rinse suyu iletkenlik < 5 uS/cm (18.2 MOhm-cm DI hedef)

**Asama B - Kademeli Piroliz**
- Ramp-1: RT -> 300 C, 2.0 C/dk, N2 250 sccm
- Hold-1: 300 C, 45 dk (nem/ucucu uzaklastirma)
- Ramp-2: 300 -> 700 C, 3.0 C/dk, N2 300 sccm
- Hold-2: 700 C, 60 dk (karbonizasyon ana penceresi)
- Ramp-3: 700 -> 860 C, 1.5 C/dk, N2 350 sccm
- Hold-3: 860 C, 35 dk (gözeneklilik/iletkenlik dengeleme)

**Asama C - Purge + Kontrollu Sogutma**
- Purge: 860 C'de 15 dk, N2 400 sccm
- Cool-down: 860 -> 200 C, <=2.5 C/dk, N2 250 sccm
- 200 C altinda numune alma

**Asama D - Rinse/Conditioning**
- 3 kademeli rinse: 40 C DI su, 15+15+20 dk (hafif karistirma)
- Opsiyonel son rinse: 20% EtOH + DI (1:4), 10 dk (kuruma hizlandirma)
- Kurutma: 110 C, 8 saat, N2 blanket

**Asama E - Karar Kapilari (Go/No-Go)**
- GC off-gas: CH4 slip proxy > baz+0.3 ise Hold-2 +15 dk
- FTIR N-bandi (veya N-retansiyon proxy) step bazinda >%3 duserse pik 860 -> 830 C cek
- Rinse cikis iletkenligi 3. yikamada >150 uS/cm ise fallback takvime gec

### 4.2 Fallback Takvim (Utility/su riski artarsa)
- Pik sicaklik: 860 -> 820 C
- Hold-2: 60 -> 90 dk
- Hold-3: 35 -> 20 dk
- Ramp hizlari: %20 yavaslat
- Rinse: acik hat yerine kapali devre DI loop + 0.2 um filtre
- Kurutma: 90 C vakum + N2 (gece boyunca)

Neden fallback:
- N-retansiyon proxy dusus trendine karsi termal stresi azaltir.
- Su-iyon ve kontaminasyon riskinde rinse degiskenligini sinirlar.

## 5) Kompakt Uygulanabilir Deney Protokolu (Bench-Scale)
1. Numune hazirla: precursor/karbon feedstock homojenizasyonu, nem kontrolu (<%40 RH hedef).
2. TGA/mini-on deneme ile ucucu penceresini kontrol et (istege bagli ama onerilir).
3. Tube furnace'e kuvars boat ile yukle; N2 hattini leak-test et.
4. Onerilen staged pyrolysis programini uygula (A-B-C asamalari).
5. Off-gas GC numunelemesini Hold-2 ve Hold-3 sonunda yap.
6. Sogutma sonrasi kontrollu rinse/conditioning uygula.
7. QC paketi:
   - Gravimetrik yield (%)
   - Iletkenlik (4-point probe)
   - FTIR (N-fonksiyonel grup proxy)
   - Ion kromatografi (rinsate nitrate/chloride)
8. Atik yonetimi:
   - Asitli/iyonlu rinse atigini ayri topla
   - Etiket/manifest uyumu ile bertaraf et
9. Batch kabul:
   - Yield, iletkenlik, FTIR proxy ve rinse cikis iletkenligi esiklerini ayni anda sagla.

## 6) Artifact Ciktilari (Inceleme Icin)
Calisma JSON ozeti:
- `/Users/farukalpay/Documents/GitHub/hormuz-tectonochemical-engine/results/reports/materials_experiment_mcp_run_2026-04-11.json`

Model/forecast/optimizasyon dosyalari:
- `/Users/farukalpay/Documents/GitHub/hormuz-tectonochemical-engine/results/model_metrics_lb12_hz1_u128-96-64.json`
- `/Users/farukalpay/Documents/GitHub/hormuz-tectonochemical-engine/results/holdout_predictions_lb12_hz1_u128-96-64.csv`
- `/Users/farukalpay/Documents/GitHub/hormuz-tectonochemical-engine/results/horizon_forecast_lb12_hz1_u128-96-64.json`
- `/Users/farukalpay/Documents/GitHub/hormuz-tectonochemical-engine/results/optimization_summary_lb12_hz1_u128-96-64.json`

Yayinlanan MCP artifact referanslari (relative URL):
- `/mcp/artifacts/20260411T025253Z_write_artifacts_2bc2a836493b4ab49ff82b400d31645f/artifact_index.json`
- `/mcp/artifacts/20260411T025147Z_optimize_schedule_9edc4688eeef45eebe5139a7d12db169/artifact_index.json`
- `/mcp/artifacts/20260411T025045Z_forecast_observables_87add252384f4f73846707f014f86cf2/artifact_index.json`

## 7) Makine-Okunur Ek (JSON)
```json
{
  "actions_taken": [
    {
      "tool": "backend_status",
      "request_id": "c6d17184666343b883e07037275c5c2b",
      "result": "ok",
      "resolved_device": "/GPU:0"
    },
    {
      "tool": "alignment_manifest",
      "request_id": "f99bc7d9676149d8bfe2530b2cc9ffed",
      "result": "ok"
    },
    {
      "tool": "record_operational_evidence",
      "request_id": "73b110b0995c46f889358373f0880b6e",
      "result": "ok",
      "evidence_count": 7,
      "record_path": "/Users/farukalpay/Documents/GitHub/hormuz-tectonochemical-engine/results/evidence/20260411T025016026730Z_f6e75cda8ca148a6b0e90021fc97c283.json"
    },
    {
      "tool": "train_model",
      "request_id": "4662a8f2bfad4124b24faf33c7f84261",
      "result": "ok",
      "epochs_completed": 11,
      "mean_test_mae": 3.25728178024292
    },
    {
      "tool": "forecast_observables",
      "request_id": "87add252384f4f73846707f014f86cf2",
      "result": "ok",
      "steps": 6
    },
    {
      "tool": "optimize_schedule",
      "request_id": "9edc4688eeef45eebe5139a7d12db169",
      "result": "ok"
    },
    {
      "tool": "validation_protocols",
      "request_id": "c3e51b0166a642988510a616f4542a16",
      "result": "ok",
      "protocol_count": 4
    },
    {
      "tool": "write_artifacts",
      "request_id": "2bc2a836493b4ab49ff82b400d31645f",
      "result": "ok"
    }
  ],
  "assumptions": [
    "Model hedefleri elektrot-ozel degil; gaz/N/su davranisi proxy olarak yorumlandi.",
    "N2 inert gaz ve kontrollu purge altyapisi mevcut.",
    "Rinse suyu icin DI kalite kapisi uygulanabilir.",
    "Lokal utility dalgalanmasi olasiligi nedeniyle schedule robust secildi."
  ],
  "final_schedule_parameters": {
    "recommended": {
      "ramp_1": {"from_c": 25, "to_c": 300, "rate_c_per_min": 2.0, "n2_sccm": 250, "hold_min": 45},
      "ramp_2": {"from_c": 300, "to_c": 700, "rate_c_per_min": 3.0, "n2_sccm": 300, "hold_min": 60},
      "ramp_3": {"from_c": 700, "to_c": 860, "rate_c_per_min": 1.5, "n2_sccm": 350, "hold_min": 35},
      "purge": {"temp_c": 860, "duration_min": 15, "n2_sccm": 400},
      "cooldown": {"to_c": 200, "max_rate_c_per_min": 2.5, "n2_sccm": 250},
      "rinse": {"steps": 3, "temp_c": 40, "durations_min": [15, 15, 20], "exit_conductivity_uScm_target": 150},
      "conditioning": {"dry_temp_c": 110, "duration_h": 8}
    },
    "fallback": {
      "peak_temp_c": 820,
      "hold_700c_min": 90,
      "hold_peak_min": 20,
      "ramp_slowdown_factor": 1.2,
      "closed_loop_rinse": true,
      "vacuum_dry_temp_c": 90
    }
  }
}
```

## Kritik Residual Uzlasi
- **Yapisal uzlasi (acik):** Mevcut MCP model/dataset katmani elektroda ozel degiskenlerle hizalanmis degil; bu raporda proxy-eslestirme kullanildi.
- Bu nedenle bu sonuc, operasyonel karar destegi icin uygundur; fakat elektrot urun-spec optimizasyonu icin bir sonraki adimda elektrot-ozel hedef setiyle yeniden egitim gereklidir.
