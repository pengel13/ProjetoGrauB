import cv2
import numpy as np
import os


def load_stickers(folder="trabGraub/stickers"):
    stickers = []
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(
            f"Pasta '{folder}' criada. Adicione stickers (imagens PNG com transparência) nesta pasta."
        )
        return stickers
    for file in os.listdir(folder):
        if file.endswith(".png"):
            sticker = cv2.imread(os.path.join(folder, file), cv2.IMREAD_UNCHANGED)
            if sticker is not None:
                stickers.append(sticker)
    if not stickers:
        print(f"Nenhum sticker encontrado na pasta '{folder}'.")
    return stickers


def apply_filter(image, filter_type):
    filters = {
        "grayscale": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        "sepia": lambda img: cv2.transform(
            img,
            np.array(
                [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
            ),
        ),
        "invert": lambda img: cv2.bitwise_not(img),
        "blur": lambda img: cv2.GaussianBlur(img, (15, 15), 0),
        "threshold": lambda img: cv2.threshold(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY
        )[1],
        "cartoon": lambda img: cv2.bitwise_and(cv2.medianBlur(img, 7), img),
        "edges": lambda img: cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200),
        "saturate": lambda img: cv2.convertScaleAbs(img, alpha=1.2, beta=0),
        "pixelate": lambda img: cv2.resize(
            cv2.resize(img, (20, 20)), (img.shape[1], img.shape[0])
        ),
        "sharpen": lambda img: cv2.filter2D(
            img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        ),
    }
    if filter_type in filters:
        return filters[filter_type](image)
    print(f"Filtro '{filter_type}' inválido. Nenhuma modificação feita.")
    return image


def overlay_sticker(base_image, sticker, x, y):
    sticker_h, sticker_w, sticker_c = sticker.shape
    if sticker_c != 4:
        print("Sticker inválido (deve ter 4 canais, incluindo transparência).")
        return base_image

    y1, y2 = y, y + sticker_h
    x1, x2 = x, x + sticker_w

    if y2 > base_image.shape[0] or x2 > base_image.shape[1]:
        print("Sticker fora dos limites da imagem.")
        return base_image

    sticker_rgb = sticker[:, :, :3]
    sticker_alpha = sticker[:, :, 3] / 255.0

    for c in range(3):
        base_image[y1:y2, x1:x2, c] = (
            sticker_alpha * sticker_rgb[:, :, c]
            + (1 - sticker_alpha) * base_image[y1:y2, x1:x2, c]
        )

    return base_image


def save_image(image):
    save_path = input(
        "Digite o nome e local para salvar a imagem (com extensão, ex: imagem.png): "
    ).strip()
    if not save_path:
        save_path = "edited_image.png"
    cv2.imwrite(save_path, image)
    print(f"Imagem salva como '{save_path}'.")


def main():
    stickers = load_stickers()
    if not stickers:
        print("Nenhum sticker disponível. Saindo...")
        return

    print("Bem-vindo ao editor de imagens!")
    print("1. Carregar imagem")
    print("2. Capturar frame da webcam")
    choice = input("Escolha uma opção (1 ou 2): ")

    if choice == "1":
        file_path = input("Digite o caminho do arquivo a ser carregado: ").strip()
        image = cv2.imread(file_path)
        if image is None:
            print("Erro ao carregar a imagem.")
            return
    elif choice == "2":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erro ao acessar a webcam.")
            return
        ret, frame = cap.read()
        cap.release()
        if ret:
            image = frame
        else:
            print("Erro ao capturar frame.")
            return
    else:
        print("Opção inválida.")
        return

    print("\nComandos disponíveis:")
    print("[S] Salvar imagem")
    print("[F] Aplicar filtro")
    print("[T] Adicionar sticker")
    print("[Q] Sair")

    while True:
        cv2.imshow("Editor de Imagem", image)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("s"):
            save_image(image)
        elif key == ord("f"):
            print(
                "Filtros disponíveis: grayscale, sepia, invert, blur, threshold, cartoon, edges, saturate, pixelate, sharpen"
            )
            filter_type = input("Escolha um filtro: ").strip().lower()
            temp_image = apply_filter(image, filter_type)
            if temp_image is not None:
                image = temp_image
                print(f"Filtro '{filter_type}' aplicado com sucesso!")
        elif key == ord("t"):
            print("Selecione o sticker desejado:")
            for i, sticker in enumerate(stickers):
                print(f"{i + 1}. Sticker {i + 1}")
            try:
                sticker_idx = int(input("Digite o número do sticker: ")) - 1
                if 0 <= sticker_idx < len(stickers):
                    print("Clique na imagem para posicionar o sticker.")
                    x, y, _, _ = cv2.selectROI(
                        "Editor de Imagem", image, fromCenter=False, showCrosshair=True
                    )
                    image = overlay_sticker(
                        image, stickers[sticker_idx], int(x), int(y)
                    )
                    print("Sticker adicionado com sucesso!")
                else:
                    print("Sticker inválido.")
            except ValueError:
                print("Entrada inválida.")
        elif key == ord("q"):
            print("Saindo...")
            break
        else:
            print("Comando inválido. Tente novamente.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
