import sys
import os
import unicodedata
import cv2
import threading
import time
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QLineEdit,
    QMessageBox, QComboBox, QInputDialog, QHBoxLayout
)
from PyQt6.QtCore import QTimer, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap, QFont, QBrush, QPalette
from deepface import DeepFace
from scipy.spatial.distance import cosine

DB_PATH = "usuarios"
THRESHOLD = 0.40
SECURITY_KEY = "123456"

# ============================
# Funções auxiliares
# ============================

def normalizar_nome(texto):
    texto = ''.join(
        c for c in unicodedata.normalize('NFKD', texto)
        if not unicodedata.combining(c)
    )
    texto = texto.replace(" ", "_").lower()
    return texto


def cadastrar_usuario(nome, cargo):
    cargo_norm = normalizar_nome(cargo)
    pasta_cargo = os.path.join(DB_PATH, cargo_norm)
    os.makedirs(pasta_cargo, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        QMessageBox.warning(None, "Erro", "Não foi possível abrir a câmera.")
        return False

    QMessageBox.information(None, "Instrução", "Posicione seu rosto e pressione 's' para salvar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Cadastro de Usuário", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            caminho_foto = os.path.join(pasta_cargo, f"{nome}.jpg")
            cv2.imwrite(caminho_foto, frame)
            QMessageBox.information(None, "Sucesso", f"Foto salva em: {caminho_foto}")
            break
        elif key == ord('q'):
            QMessageBox.information(None, "Cancelado", "Cadastro cancelado.")
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()
    return True


def carregar_base_embeddings():
    embeddings = []
    if not os.path.exists(DB_PATH):
        return embeddings

    for root, dirs, files in os.walk(DB_PATH):
        for file in files:
            caminho = os.path.join(root, file)
            nome = os.path.splitext(os.path.basename(caminho))[0]
            nivel = os.path.basename(root)
            try:
                emb = DeepFace.represent(caminho, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
                embeddings.append({"embedding": emb, "nome": nome, "Nível": nivel})
            except Exception as e:
                print(f"Erro ao processar {caminho}: {e}")
    return embeddings


# ============================
# DASHBOARD (Imagem conforme nível)
# ============================

class Dashboard(QWidget):
    def __init__(self, nivel="nivel_1", nome=""):
        super().__init__()
        self.nivel = nivel
        self.nome = nome.capitalize()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        # Texto de boas vindas
        titulo = QLabel(f"Bem vindo, {self.nome}!")
        titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        titulo.setStyleSheet("""
            color: #00FF88;
            font-size: 32px;
            font-weight: bold;
        """)
        layout.addWidget(titulo, alignment=Qt.AlignmentFlag.AlignCenter)

        # Imagem centralizada e ajustada (sem botão de fechar)
        self.lbl_imagem = QLabel()
        self.lbl_imagem.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_imagem, alignment=Qt.AlignmentFlag.AlignCenter)

        # Botão de sair do dashboard
        btn_sair = QPushButton("Encerrar Sessão")
        btn_sair.clicked.connect(self.voltar_login)
        btn_sair.setStyleSheet("""
            QPushButton {
                background-color: #00FF88;
                color: #FFFFFF; /* Cor do texto alterada para branco */
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #00CC70;
            }
        """)
        btn_sair.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(btn_sair, alignment=Qt.AlignmentFlag.AlignCenter)

        self.carregar_imagem_enquadrada()

    def carregar_imagem_enquadrada(self):
        """Carrega a imagem do nível centralizada e em tamanho reduzido."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        caminho = os.path.join(base_dir, "imagens", f"{self.nivel}.png")

        if not os.path.exists(caminho):
            self.lbl_imagem.setText(f"Imagem não encontrada:\n{caminho}")
            self.lbl_imagem.setStyleSheet("color: white; font-size: 16px;")
            return

        pixmap = QPixmap(caminho)
        screen_rect = QApplication.primaryScreen().availableGeometry()

        # Reduz o tamanho da imagem (ex: 70% da tela)
        largura = int(screen_rect.width() * 0.7)
        altura = int(screen_rect.height() * 0.7)

        self.lbl_imagem.setPixmap(
            pixmap.scaled(
                largura,
                altura,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def voltar_login(self):
        self.setVisible(False)
        self.parent().login_widget.setVisible(True)


# ============================
# SISTEMA PRINCIPAL
# ============================

class FaceApp(QWidget):
    atualizar_frame_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Reconhecimento Facial")
        self.setGeometry(100, 100, 900, 600)

        self.base_embeddings = carregar_base_embeddings()
        self.reconhecendo = False
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.atualizar_frame)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.atualizar_frame_signal.connect(self.atualizar_frame_reconhecido)
        self.last_process_time = 0
        self.init_ui()

    def resize_background(self, event):
        # Redimensiona a imagem de fundo quando a janela é redimensionada
        caminho_imagem = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagens", "login.png")
        pixmap = QPixmap(caminho_imagem)
        if not pixmap.isNull():
            palette = self.palette()
            palette.setBrush(self.backgroundRole(), QBrush(pixmap.scaled(event.size(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)))
            self.setPalette(palette)
        super().resizeEvent(event)

    def init_ui(self):
        # Configuração da imagem de fundo via QPalette (abordagem recomendada para PyQt)
        caminho_imagem = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagens", "login.png")
        pixmap = QPixmap(caminho_imagem)
        if not pixmap.isNull():
            palette = self.palette()
            palette.setBrush(self.backgroundRole(), QBrush(pixmap.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)))
            self.setPalette(palette)
            self.setAutoFillBackground(True)
        
        self.setStyleSheet("""
            QWidget { background-color: transparent; color: #E0E0E0; font-family: 'Segoe UI'; }
            QLabel { color: #00FF88; }
        """)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Conecta o evento de redimensionamento para garantir que a imagem de fundo seja redimensionada
        self.resizeEvent = self.resize_background

        # Botão de fechar (❌)
        top_bar = QHBoxLayout()
        btn_fechar = QPushButton("❌")
        btn_fechar.setFixedSize(45, 45)
        btn_fechar.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: red;
                border: none;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #FF5555;
            }
        """)
        btn_fechar.clicked.connect(self.close)
        btn_fechar.setCursor(Qt.CursorShape.PointingHandCursor)
        top_bar.addWidget(btn_fechar)
        top_bar.addStretch()
        self.layout.addLayout(top_bar)
        
        
        # ==== LOGIN ====
        self.login_widget = QWidget()
        login_layout = QVBoxLayout()
        login_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.login_widget.setLayout(login_layout)

        self.lbl_titulo = QLabel("COFRE DO MINISTÉRIO DO MEIO AMBIENTE")
        font = QFont()
        font.setPointSize(28)
        font.setBold(True)
        self.lbl_titulo.setFont(font)
        self.lbl_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        login_layout.addWidget(self.lbl_titulo)

        login_layout.addSpacing(60)

        self.btn_login = QPushButton("ENTRAR COM RECONHECIMENTO FACIAL 🔓")
        self.btn_login.clicked.connect(self.login_facial)
        self.btn_login.setStyleSheet("""
            QPushButton {
                background-color: #00FF88;
                color: #FFFFFF;
                border-radius: 8px;
                padding: 12px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #00CC70;
            }
        """)
        login_layout.addWidget(self.btn_login)

        self.btn_cadastrar = QPushButton("CADASTRAR USUÁRIO ✏️")
        self.btn_cadastrar.clicked.connect(self.solicitar_chave)
        self.btn_cadastrar.setStyleSheet("""
            QPushButton {
                background-color: #00FF88;
                color: #FFFFFF;
                border-radius: 8px;
                padding: 12px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #00CC70;
            }
        """)
        login_layout.addWidget(self.btn_cadastrar)

        self.btn_remover = QPushButton("APAGAR USUÁRIO 🗑️")
        self.btn_remover.clicked.connect(self.apagar_usuario)
        self.btn_remover.setStyleSheet("""
            QPushButton {
                background-color: #00FF88;
                color: #FFFFFF;
                border-radius: 8px;
                padding: 12px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #00CC70;
            }
        """)
        login_layout.addWidget(self.btn_remover)

        self.btn_login.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cadastrar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_remover.setCursor(Qt.CursorShape.PointingHandCursor)

        self.layout.addWidget(self.login_widget)

        # ==== CADASTRO ====
        self.cadastro_widget = QWidget()
        cadastro_layout = QVBoxLayout()
        cadastro_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cadastro_widget.setLayout(cadastro_layout)
        self.cadastro_widget.setVisible(False)

        self.input_nome_cad = QLineEdit()
        self.input_nome_cad.setPlaceholderText("Nome do usuário")
        self.input_nome_cad.setFixedHeight(50)
        font_input = QFont()
        font_input.setPointSize(16)
        self.input_nome_cad.setFont(font_input)
        cadastro_layout.addWidget(self.input_nome_cad)

        self.select_cargo = QComboBox()
        self.select_cargo.addItems(["Nível 1", "Nível 2", "Nível 3"])
        self.select_cargo.setFixedHeight(50)
        self.select_cargo.setFont(font_input)
        self.select_cargo.setStyleSheet("""
            QComboBox {
                background-color: rgba(0, 0, 0, 0);
                color: white;
                border: 2px solid;
                border-radius: 8px;
                padding: 8px;
            }
            QComboBox::drop-down {
                background-color: rgba(0, 0, 0, 0);
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(0, 0, 0, 180); /* semi-transparente na lista */
                color: white;
                selection-background-color: #00FF88;
            }
        """)
        cadastro_layout.addWidget(self.select_cargo)

        btn_confirmar = QPushButton("Cadastrar")
        btn_confirmar.clicked.connect(self.cadastrar)
        cadastro_layout.addWidget(btn_confirmar)

        btn_voltar = QPushButton("Voltar")
        btn_voltar.clicked.connect(self.voltar_login)
        cadastro_layout.addWidget(btn_voltar)

        btn_confirmar.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_voltar.setCursor(Qt.CursorShape.PointingHandCursor)

        self.layout.addWidget(self.cadastro_widget)

        # ==== RECONHECIMENTO ====
        self.recon_widget = QWidget()
        recon_layout = QVBoxLayout()
        recon_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recon_widget.setLayout(recon_layout)
        self.recon_widget.setVisible(False)

        self.lbl_bemvindo = QLabel("Reconhecendo rosto...")
        font2 = QFont()
        font2.setPointSize(18)
        font2.setBold(True)
        self.lbl_bemvindo.setFont(font2)
        recon_layout.addWidget(self.lbl_bemvindo)

        self.label_video = QLabel()
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        recon_layout.addWidget(self.label_video)

        btn_parar = QPushButton("Encerrar Sessão")
        btn_parar.clicked.connect(self.sair_reconhecimento)
        btn_parar.setCursor(Qt.CursorShape.PointingHandCursor)
        recon_layout.addWidget(btn_parar)

        self.layout.addWidget(self.recon_widget)

        # Garante que os widgets internos sejam transparentes para que a imagem de fundo apareça
        self.login_widget.setStyleSheet("background-color: transparent;")
        self.cadastro_widget.setStyleSheet("background-color: transparent;")
        self.recon_widget.setStyleSheet("background-color: transparent;")

        self.dashboard_widget = None

    # ======================
    # Lógica principal
    # ======================

    def solicitar_chave(self):
        chave, ok = QInputDialog.getText(self, "Autorização", "Digite a chave de segurança:", QLineEdit.EchoMode.Password)
        if ok:
            if chave == SECURITY_KEY:
                self.mostrar_cadastro()
            else:
                QMessageBox.warning(self, "Erro", "Chave inválida!")

    def mostrar_cadastro(self):
        self.login_widget.setVisible(False)
        self.cadastro_widget.setVisible(True)

    def voltar_login(self):
        self.cadastro_widget.setVisible(False)
        self.login_widget.setVisible(True)

    def apagar_usuario(self):
        chave, ok = QInputDialog.getText(self, "Autorização", "Digite a chave de segurança:", QLineEdit.EchoMode.Password)
        if not ok:
            return
        if chave != SECURITY_KEY:
            QMessageBox.warning(self, "Erro", "Chave de segurança inválida!")
            return

        niveis = ["nivel_1", "nivel_2", "nivel_3"]
        total_usuarios = 0
        for nivel in niveis:
            pasta_nivel = os.path.join(DB_PATH, nivel)
            if os.path.exists(pasta_nivel):
                total_usuarios += len([f for f in os.listdir(pasta_nivel)
                                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        if total_usuarios == 0:
            QMessageBox.information(self, "Nenhum Usuário", "Não há usuários cadastrados.")
            return

        dialog = QInputDialog(self)
        dialog.setWindowTitle("Selecionar Nível")
        dialog.setLabelText("Escolha o nível:")
        dialog.setComboBoxItems(["Nível 1", "Nível 2", "Nível 3"])

        # 🔹 Torna o fundo da janela translúcido
        dialog.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # 🔹 Aplica estilo transparente aos widgets internos
        dialog.setStyleSheet("""
            QDialog, QInputDialog {
                background-color: rgba(0, 0, 0, 0);
            }
            QLabel {
                color: white;
            }
            QComboBox, QLineEdit {
                background-color: rgba(0, 0, 0, 120);
                color: white;
                border: 2px solid;
                border-radius: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(0, 0, 0, 200);
                color: white;
                selection-background-color: #00FF88;
            }
            QPushButton {
                background-color: #00FF88;
                color: white;
                border-radius: 6px;
                padding: 6px 10px;
            }
            QPushButton:hover {
                background-color: #00CC70;
            }
        """)
        ok1 = dialog.exec()
        cargo = dialog.textValue()
        if not ok1:
            return

        cargo_norm = normalizar_nome(cargo)
        pasta_cargo = os.path.join(DB_PATH, cargo_norm)
        if not os.path.exists(pasta_cargo):
            QMessageBox.information(self, "Sem usuários", f"Não há usuários no {cargo}.")
            return

        usuarios = [os.path.splitext(f)[0] for f in os.listdir(pasta_cargo)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not usuarios:
            QMessageBox.information(self, "Sem usuários", f"Não há usuários no {cargo}.")
            return

        nome, ok2 = QInputDialog.getItem(self, "Remover Usuário", f"Selecione o usuário a apagar do {cargo}:",
                                         usuarios, 0, False)
        if not ok2:
            return

        caminho_foto = os.path.join(pasta_cargo, f"{nome}.jpg")
        if os.path.exists(caminho_foto):
            os.remove(caminho_foto)
            self.base_embeddings = carregar_base_embeddings()
            QMessageBox.information(self, "Sucesso", f"Usuário '{nome}' removido.")
        else:
            QMessageBox.warning(self, "Erro", f"O arquivo '{nome}.jpg' não foi encontrado.")

    def cadastrar(self):
        nome = self.input_nome_cad.text().strip()
        cargo = self.select_cargo.currentText().strip()
        if not nome or not cargo:
            QMessageBox.warning(self, "Erro", "Preencha todos os campos!")
            return
        if cadastrar_usuario(nome, cargo):
            self.base_embeddings = carregar_base_embeddings()
            QMessageBox.information(self, "Sucesso", f"Usuário {nome} cadastrado!")
            self.voltar_login()

    def login_facial(self):
        if not self.base_embeddings:
            QMessageBox.warning(self, "Atenção", "Nenhum usuário cadastrado.")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Erro", "Não foi possível abrir a câmera.")
            return
        self.reconhecendo = True
        self.timer.start(30)
        self.login_widget.setVisible(False)
        self.recon_widget.setVisible(True)
        self.lbl_bemvindo.setText("Reconhecendo rosto...")

    def atualizar_frame(self):
        if not self.reconhecendo or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        if faces is not None and len(faces) > 0 and (time.time() - self.last_process_time > 2):
            self.last_process_time = time.time()
            (x, y, w, h) = faces[0]
            roi = frame[y:y + h, x:x + w]
            threading.Thread(target=self.processar_face_thread, args=(roi,), daemon=True).start()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(qt_img))

    def processar_face_thread(self, roi):
        try:
            emb_frame = DeepFace.represent(roi, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
        except:
            nome, nivel = "Desconhecido", ""
        else:
            nome, nivel = "Desconhecido", ""
            min_dist = THRESHOLD
            for user in self.base_embeddings:
                dist = cosine(emb_frame, user["embedding"])
                if dist < min_dist:
                    nome, nivel = user["nome"], user["Nível"]
                    break
        self.atualizar_frame_signal.emit({"nome": nome, "Nível": nivel})

    def atualizar_frame_reconhecido(self, data):
        nome, nivel = data["nome"], data["Nível"]
        self.parar_reconhecimento()
        if nome != "Desconhecido":
            self.recon_widget.setVisible(False)
            nivel_norm = normalizar_nome(nivel)
            if self.dashboard_widget:
                self.dashboard_widget.setParent(None)
            self.dashboard_widget = Dashboard(nivel_norm, nome)
            self.layout.addWidget(self.dashboard_widget)
            self.dashboard_widget.showFullScreen()
        else:
            QMessageBox.warning(self, "Acesso Negado", "Você não tem acesso.")
            self.recon_widget.setVisible(False)
            self.login_widget.setVisible(True)

    def parar_reconhecimento(self):
        self.reconhecendo = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()

    def sair_reconhecimento(self):
        self.parar_reconhecimento()
        self.recon_widget.setVisible(False)
        self.login_widget.setVisible(True)

    def closeEvent(self, event):
        self.parar_reconhecimento()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()


# ============================
# Execução principal
# ============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    janela = FaceApp()
    janela.showFullScreen()
    sys.exit(app.exec())
