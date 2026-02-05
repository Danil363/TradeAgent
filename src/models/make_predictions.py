import numpy as np
import torch


class PricePredictor:
    """
    Класс, который загружает модели предсказания цены
    и делает прогнозы t+1, t+3, t+8.
    """

    def __init__(self,
                 model_t1_path="models/AAPL/model_t1.pth",
                 model_t3_path="models/AAPL/model_t3.pth",
                 model_t8_path="models/AAPL/model_t8.pth",
                 device="cpu"):

        self.device = device

        # Загружаем модели
        self.model_t1 = torch.load(model_t1_path, map_location=device)
        self.model_t3 = torch.load(model_t3_path, map_location=device)
        self.model_t8 = torch.load(model_t8_path, map_location=device)

        self.model_t1.eval()
        self.model_t3.eval()
        self.model_t8.eval()

        print("📌 Загружены модели t+1, t+3, t+8")

    def _prepare_input(self, prices_window):
        """
        Преобразует окно цен в формат для PyTorch модели.
        """
        arr = np.array(prices_window, dtype=np.float32)
        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(-1)
        return tensor.to(self.device)

    def predict(self, prices_window):
        """
        Делает прогнозы t+1, t+3, t+8
        """

        x = self._prepare_input(prices_window)

        with torch.no_grad():
            p1 = self.model_t1(x).item()
            p3 = self.model_t3(x).item()
            p8 = self.model_t8(x).item()

        return p1, p3, p8


# ============================================================
#   Удобная функция для вызова из любой программы
# ============================================================

def make_price_predictions(prices_window,
                           model_t1="models/AAPL/model_t1.pth",
                           model_t3="models/AAPL/model_t3.pth",
                           model_t8="models/AAPL/model_t8.pth"):

    """
    Универсальная функция для создания прогнозов.
    На вход:
        prices_window — список последних цен (например 60)
    На выход:
        pred_t1, pred_t3, pred_t8
    """

    predictor = PricePredictor(
        model_t1_path=model_t1,
        model_t3_path=model_t3,
        model_t8_path=model_t8
    )

    return predictor.predict(prices_window)
