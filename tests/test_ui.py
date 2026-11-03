"""The few custom pieces of the app's look (risk strip, verdict)."""
from app import ui


def test_risk_strip_places_the_marker_and_the_call_line():
    h = ui.risk_strip(92.6, 80.6, 0.65, "Highest")
    assert 'class="rs-mark" style="left:92.6%"' in h
    assert 'style="left:80.6%"' in h and "calls start at 0.65" in h
    assert '<span class="on">Highest</span>' in h
    assert "riskier than 93% of customers" in h


def test_risk_strip_clamps_out_of_range_and_copes_without_a_threshold():
    h = ui.risk_strip(140, None, 0.65, "Highest")
    assert "left:100.0%" in h and "rs-line" not in h


def test_verdict_wording():
    assert "Call this customer" in ui.verdict(True, 0.88, 0.65)
    assert "No call needed" in ui.verdict(False, 0.11, 0.65)


def test_drivers_chart_colours_follow_direction():
    drivers = [{"label": "Age", "value": "45", "impact": 0.7}, {"label": "Gender", "value": "Male", "impact": -0.2}]
    fig = ui.drivers_chart(drivers)
    colours = list(fig.data[0].marker.color)
    assert colours == [ui.COOL, ui.HEAT]          # reversed order: bottom bar first
