import streamlit as st
import numpy as np
import pandas as pd
import random
from typing import Tuple

def generate_integer_matrix_with_target_average(rows: int, cols: int, target_avg: float,
                                                min_val: int = 0, max_val: int = 3) -> np.ndarray:
    total_elements = rows * cols
    target_sum = target_avg * total_elements

    matrix = np.full((rows, cols), min_val, dtype=int)
    current_sum = np.sum(matrix)
    remaining_sum = target_sum - current_sum

    positions = [(i, j) for i in range(rows) for j in range(cols)]
    random.shuffle(positions)

    pos_index = 0
    while remaining_sum > 0 and pos_index < len(positions):
        i, j = positions[pos_index % len(positions)]
        if matrix[i, j] < max_val and remaining_sum >= 1:
            increment = min(max_val - matrix[i, j], int(remaining_sum))
            matrix[i, j] += increment
            remaining_sum -= increment
        pos_index += 1
        if pos_index % total_elements == 0:
            random.shuffle(positions)

    max_passes = 10
    current_pass = 0
    while remaining_sum > 0.5 and current_pass < max_passes:
        random.shuffle(positions)
        for i, j in positions:
            if matrix[i, j] < max_val and remaining_sum > 0.5:
                matrix[i, j] += 1
                remaining_sum -= 1
                if remaining_sum <= 0.5:
                    break
        current_pass += 1

    return matrix

def calculate_possible_average_range(rows: int, cols: int, min_val: int = 0, max_val: int = 3) -> Tuple[float, float]:
    return float(min_val), float(max_val)

def main():
    st.set_page_config(page_title="Integer Matrix Generator", page_icon="🔢", layout="wide")

    st.title("🔢 Integer Matrix Generator")
    st.markdown("Generate **integer-only matrices** (0–3) with a target average value.")

    with st.sidebar:
        st.header("Matrix Parameters")
        rows = st.number_input("Rows", min_value=1, max_value=20, value=4, step=1)
        cols = st.number_input("Columns", min_value=1, max_value=20, value=16, step=1)

        min_possible, max_possible = calculate_possible_average_range(rows, cols)

        st.subheader("Average Range")
        st.info(f"Allowed: {min_possible:.1f} – {max_possible:.1f}")

        min_avg_range = st.number_input(
            "Minimum Average",
            min_value=min_possible,
            max_value=max_possible,
            value=2.0,
            step=0.1,
            format="%.1f"
        )

        max_avg_range = st.number_input(
            "Maximum Average",
            min_value=min_avg_range,
            max_value=max_possible,
            value=2.5,
            step=0.1,
            format="%.1f"
        )

        if min_avg_range >= max_avg_range:
            st.error("⚠️ Max average must be greater than min average!")
            return

        if st.button("🎲 Generate Matrix"):
            target_avg = random.uniform(min_avg_range, max_avg_range)
            matrix = generate_integer_matrix_with_target_average(rows, cols, target_avg)
            st.session_state.matrix = matrix
            st.session_state.target_avg = target_avg

    if "matrix" in st.session_state:
        matrix = st.session_state.matrix
        target_avg = st.session_state.target_avg
        actual_avg = np.mean(matrix)

        st.subheader("🎯 Generated Matrix")
        df = pd.DataFrame(matrix)
        df.columns = [f"Col {i+1}" for i in range(df.shape[1])]
        df.index = [f"Row {i+1}" for i in range(df.shape[0])]

        def color_matrix(val):
            if val == 0:
                return 'background-color: #ffebee; color: #c62828; font-weight: bold'
            elif val == 1:
                return 'background-color: #fff3e0; color: #ef6c00; font-weight: bold'
            elif val == 2:
                return 'background-color: #f3e5f5; color: #7b1fa2; font-weight: bold'
            elif val == 3:
                return 'background-color: #e8f5e9; color: #2e7d32; font-weight: bold'
            return ''

        styled_df = df.style.applymap(color_matrix).format("{:.0f}")
        st.dataframe(styled_df, use_container_width=True)

        st.subheader("📊 Matrix Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Target Avg", f"{target_avg:.3f}")
        col2.metric("Actual Avg", f"{actual_avg:.3f}", delta=f"{actual_avg - target_avg:.3f}")
        col3.metric("Std. Dev.", f"{np.std(matrix):.3f}")

        st.markdown("---")
        st.markdown("**Matrix Grid**")
        for i, row in enumerate(matrix):
            st.markdown(f"**Row {i+1}:** " + " | ".join(str(val) for val in row))

        st.markdown("---")
        st.subheader("📥 Download")
        csv = df.to_csv()
        st.download_button(
            label="Download Matrix as CSV",
            data=csv,
            file_name=f"matrix_{rows}x{cols}_{actual_avg:.2f}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
