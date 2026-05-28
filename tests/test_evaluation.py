from src.evaluation.portfolio_metrics import build_table_9_panel_a


def test_table_9_panel_a():
    table_9 = build_table_9_panel_a()
    assert table_9.shape[0] == 6  # 5 facteurs + 1/N
    assert "Annualized return (%)" in table_9.columns
    assert "CDB" in table_9.columns
    print("\nTABLE 9 — PANEL A\n")
    print(table_9.round(3))


if __name__ == "__main__":
    test_table_9_panel_a()