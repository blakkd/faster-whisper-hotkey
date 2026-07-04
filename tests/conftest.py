"""Pytest configuration for integration tests."""



def pytest_addoption(parser):
    parser.addoption(
        "--force-cuda",
        action="store_true",
        default=False,
        help="Force CUDA configs even when no GPU is available",
    )
