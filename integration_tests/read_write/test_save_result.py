from eddington import FittingResult

FIT_RESULT = FittingResult(
    a0=[1.0, 3.0],
    a=[1.1, 2.98],
    aerr=[0.1, 0.76],
    acov=[[0.01, 2.3], [2.3, 0.988]],
    chi2=8.276,
    degrees_of_freedom=5,
)


def test_save_result_as_text(tmpdir):
    output_path = tmpdir / "fit_result.txt"
    FIT_RESULT.save_txt(output_path)

    assert output_path.exists(), "Output text file was not written"


def test_save_result_as_json(tmpdir):
    output_path = tmpdir / "fit_result.json"
    FIT_RESULT.save_json(output_path)

    assert output_path.exists(), "Output json file was not written"
