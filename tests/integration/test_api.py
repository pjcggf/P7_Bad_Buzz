import pytest
from httpx import AsyncClient
from p7_global.api.app import app

@pytest.mark.asyncio
async def test_root_is_up():
    """
    Check si la route root renvoie bien un code 200.
    """
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_root_returns_greeting():
    """
    Check si la route root renvoie bien le message d'accueil.
    """
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.json() == f'Welcome to root, model: {app.state.model.name}'

test_params = {'text': "I love you so much, you're wonderful"}

@pytest.mark.asyncio
async def test_predict_is_up():
    """
    Check si la prédiction renvoie bien un code 200.
    """
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/predict_single_tweet", params=test_params)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_is_positive():
    """
    Check si la prédiction est correcte avec une phrase positive.
    """
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/predict_single_tweet", params=test_params)
    assert response.json() == 'Positif', f"""Renvoie un prédiction négative pour
    le texte {test_params['text']}"""
