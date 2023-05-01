from src.cluster import KMeans


def test_model_creation():
    model = KMeans(k=3, init_c=None, verbose=True)
    assert model.k == 3
    assert model.init_c is None
    assert model.verbose == True
    assert model.clusters == None