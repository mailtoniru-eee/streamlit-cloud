with tab3:
    st.header("🏆 Best Configuration for Selected Dataset")

    df3 = df[df["input_dataset"] == input_dataset]

    group_fields = [
        "vector_db", "reranking_model", "repacking_strategy", "summarization_model"
    ]

    metrics = [
        "context_relevance", "context_utilization", "adherence", "completeness",
        "hallucination_auroc", "relevance_rmse", "utilization_rmse"
    ]

    if df3.empty:
        st.warning("No data available for the selected dataset.")
    else:
        # 1. Group by config and compute mean of each metric
        grouped = df3.groupby(group_fields)[metrics].mean().reset_index()

        # 2. Normalize each metric column (Min-Max scaling)
        norm_cols = []
        for metric in metrics:
            min_val = grouped[metric].min()
            max_val = grouped[metric].max()

            if min_val == max_val:
                norm = 1.0  # constant column
            else:
                norm = (grouped[metric] - min_val) / (max_val - min_val)

            # Invert RMSE metrics (lower is better)
            if metric in ["relevance_rmse", "utilization_rmse"]:
                norm = 1 - norm

            norm_col = metric + "_norm"
            grouped[norm_col] = norm
            norm_cols.append(norm_col)

        # 3. Calculate total normalized score
        grouped["total_score"] = grouped[norm_cols].sum(axis=1)

        # 4. Get top configs
        best_configs = grouped.sort_values(by="total_score", ascending=False).head(3)

        st.markdown("### 🥇 Top Configurations Based on Normalized Scores")
        st.dataframe(best_configs[group_fields + ["total_score"] + norm_cols].style.format(precision=3), use_container_width=True)

        # Optional: Chart comparison of total score
        st.markdown("### 📊 Total Score Comparison")

        chart = (
            alt.Chart(best_configs)
            .mark_bar()
            .encode(
                x=alt.X("total_score:Q", title="Total Normalized Score"),
                y=alt.Y("summarization_model:N", title="Summarization Model"),
                color=alt.Color("vector_db:N", title="Vector DB"),
                tooltip=group_fields + ["total_score"]
            )
            .properties(height=300)
        )

        st.altair_chart(chart, use_container_width=True)
