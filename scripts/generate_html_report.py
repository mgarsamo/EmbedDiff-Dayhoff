import os
import base64
from datetime import datetime
import imghdr  # To validate image type

def generate_html_report(output_path="embeddiff_summary_report.html"):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    ordered_filenames = [
         "fig_tsne_by_domain.png",
        "fig2b_loss_train_val.png",
        "fig3a_generated_tsne.png",
        "fig5a_decoder_loss.png",
        "fig5a_real_real_cosine.png",
        "fig5b_gen_gen_cosine.png",
        "fig5c_real_gen_cosine.png",
        "fig5b_identity_histogram.png",
        "fig5c_entropy_scatter.png",
        "fig5d_all_histograms.png",
        "fig5f_tsne_domain_overlay.png"
    ]

    figures_dir = "figures"
    plots = [fname for fname in ordered_filenames if os.path.exists(os.path.join(figures_dir, fname))]
    if not plots:
        print(f"⚠️ No valid image files found in {figures_dir}. Check file paths and extensions.")
        return

    blast_csv = "data/blast_results/blast_summary_local.csv"
    identity_csv = os.path.join(figures_dir, "fig5b_identity_scores.csv")
    decoded_fasta = "data/decoded_embeddiff.fasta"

    with open(output_path, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EmbedDiff Summary Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', sans-serif;
            background-color: #f4f6f8;
            color: #333;
            margin: 0;
            padding: 2em;
        }}
        h1 {{
            font-size: 2.2em;
            color: #1e3a8a;
            margin-bottom: 0.2em;
        }}
        h2 {{
            font-size: 1.5em;
            margin-top: 2em;
            border-bottom: 2px solid #ccc;
            padding-bottom: 0.3em;
        }}
        .grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 24px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 1em;
            flex: 1 1 45%;
            max-width: 45%;
        }}
        .card img {{
            width: 100%;
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid #ddd;
            display: block;
        }}
        .figure-title {{
            font-weight: 600;
            margin-top: 0.5em;
            margin-bottom: 1em;
            font-size: 1em;
            color: #111827;
        }}
        a.download {{
            display: inline-block;
            margin: 10px 10px 0 0;
            font-weight: bold;
            color: #2563eb;
            text-decoration: none;
            border-bottom: 1px solid transparent;
        }}
        a.download:hover {{
            border-color: #2563eb;
        }}
    </style>
</head>
<body>
    <h1>🧬 EmbedDiff Summary Report</h1>
    <p><strong>Date Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <h2>📊 Visualizations</h2>
    <div class="grid">
""")

        for plot in plots:
            full_path = os.path.join(figures_dir, plot)
            # Validate image type
            img_type = imghdr.what(full_path)
            if img_type:
                title = plot.replace("fig", "Figure ").replace("_", " ").replace(".png", "").title()
                f.write(f"""
        <div class="card">
            <div class="figure-title">{title}</div>
            <img src="figures/{plot}" alt="{title}">
        </div>
""")
            else:
                print(f"⚠️ Skipping {plot} due to invalid image format.")

        f.write("""
    </div>

    <h2>📥 Downloads</h2>
""")
        if os.path.exists(blast_csv):
            f.write(f"<a class='download' href='{blast_csv}' download>Download BLAST Summary CSV</a>\n")
        if os.path.exists(identity_csv):
            f.write(f"<a class='download' href='{identity_csv}' download>Download Identity Scores CSV</a>\n")
        if os.path.exists(decoded_fasta):
            f.write(f"<a class='download' href='{decoded_fasta}' download>Download Final FASTA</a>\n")

        f.write("""
</body>
</html>
""")

    print(f"✅ Final HTML report saved to {output_path}")

if __name__ == "__main__":
    generate_html_report()