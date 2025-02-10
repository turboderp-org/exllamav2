import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QFileDialog,
    QWidget,
    QMessageBox,
)
from PyQt5.QtGui import (
    QPixmap,
    QImage,
    QTextCursor,
    QPainter,
    QPen,
    QFont,
    QColor,
    QRegion,
)
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QBuffer

from PIL import Image
import requests
from io import BytesIO
import time
import pprint

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2VisionTower,
)

from exllamav2.generator import (
    ExLlamaV2DynamicGenerator,
    ExLlamaV2DynamicJob,
    ExLlamaV2Sampler,
)

class Model:

    current_image: Image or None = None
    current_description: str

    def __init__(self, model_directory, bbox_mode: str):
        self.model_directory = model_directory
        self.config = None
        self.vision_model = None
        self.model = None
        self.cache = None
        self.tokenizer = None
        self.current_image = None
        self.current_emb = None
        self.current_description = ""
        bbox_funcs = {
            "qwen2": self.get_grounding_bb_qwen2,
            "qwen25": self.get_grounding_bb_qwen25,
        }
        self.bbox_func = bbox_funcs[bbox_mode]

    def load(self):
        """Load and initialize the things"""
        self.config = ExLlamaV2Config(self.model_directory)
        self.config.max_seq_len = 8192

        self.vision_model = ExLlamaV2VisionTower(self.config)
        self.vision_model.load(progress = True)

        self.model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache(self.model, lazy = True, max_seq_len = 32768)
        self.model.load_autosplit(self.cache, progress = True)
        self.tokenizer = ExLlamaV2Tokenizer(self.config)

        self.generator = ExLlamaV2DynamicGenerator(
            model = self.model,
            cache = self.cache,
            tokenizer = self.tokenizer,
        )

    def set_image(self, image: Image):
        w, h = image.size
        print(f"New image: {w} x {h} pixels")
        self.current_image = image
        self.current_description = ""

    def get_prompt(self):
        prompt = (
            "<|im_start|>system\n" +
            "You are a helpful assistant.<|im_end|>\n" +
            "<|im_start|>user\n" +
            self.current_emb.text_alias +
            "Describe the image in detail." +
            "\n" +
            "<|im_start|>assistant\n"
        )
        return prompt

    def inference(self, settext_fn, update_fn):
        """Run inference on the current image, stream results"""

        if self.current_image is None:
            settext_fn("No image loaded.")
            return

        settext_fn("")
        update_fn()

        self.current_emb = self.vision_model.get_image_embeddings(
            model = self.model,
            tokenizer = self.tokenizer,
            image = self.current_image,
        )

        prompt = self.get_prompt()

        input_ids = self.tokenizer.encode(
            prompt,
            add_bos = True,
            encode_special_tokens = True,
            embeddings = [self.current_emb],
        )

        job = ExLlamaV2DynamicJob(
            input_ids = input_ids,
            max_new_tokens = 1000,
            decode_special_tokens = True,
            stop_conditions = [self.tokenizer.eos_token_id],
            gen_settings = ExLlamaV2Sampler.Settings.greedy(),
            embeddings = [self.current_emb],
        )

        self.generator.enqueue(job)

        text = ""
        lastupdate = time.time()

        while self.generator.num_remaining_jobs():
            results = self.generator.iterate()
            for result in results:
                text += result.get("text", "")

            # Update at max 30 fps
            if time.time() - lastupdate > (1/30):
                lastupdate = time.time()
                settext_fn(text)
                update_fn()
#
#         text = \
# """And you may find yourself living in a shotgun shack
# And you may find yourself in another part of the world
# And you may find yourself behind the wheel of a large automobile
# And you may find yourself in a beautiful house, with a beautiful wife
# And you may ask yourself, "Well, how did I get here?\""""

        settext_fn(text)
        update_fn()
        self.current_description = text
        print("Image description from model:")
        print(text)

    def get_grounding_bb_qwen2(self, start, end) -> tuple:
        """
        Prompt the model again and try to extraxt the bounding box of the image details indicated by selected portion
        of the description. We do this by repeating the exact same prompt up to and including the selected text, but
        enclosed in the special tokens that Qwen would emit when prompted for grounding. Qwen is then strongly biased
        towards completing the bounding box.

        Since we're using the same description as the model originally generated, all keys/values for the system
        prompt, image and generated description up to the selection will be reused from the cache.
        """

        if start >= end:
            return None, None

        # Including leading space
        if start > 0 and self.current_description[start - 1] == " ":
            start -= 1

        # Repeat the same prompt up to the selection, with grounding tokens added
        prompt = self.get_prompt()
        prompt += self.current_description[:start]
        prompt += "<|object_ref_start|>"
        prompt += self.current_description[start:end]
        prompt += "<|object_ref_end|><|box_start|>("

        bb_string, res = self.generator.generate(
            prompt = prompt,
            add_bos = True,
            max_new_tokens = 25,
            stop_conditions = [self.tokenizer.single_id("<|box_end|>")],
            gen_settings = ExLlamaV2Sampler.Settings.greedy(),
            embeddings = [self.current_emb],
            completion_only = True,
            return_last_results = True,  # debug purposes
        )
        bb_string = "(" + bb_string

        print(f"Generation: {bb_string}")
        pprint.pprint(res, indent = 4)

        # BB string is in the format "(x0,y0),(x1,y1)" with integer coordinates normalized to a range of 1000x1000

        try:
            parts = bb_string.strip("()").split("),(")
            a = tuple(map(int, parts[0].split(",")))
            b = tuple(map(int, parts[1].split(",")))
            a = (a[0] / 1000.0, a[1] / 1000.0)
            b = (b[0] / 1000.0, b[1] / 1000.0)
        except:
            print("No bounding box could be determined")
            a, b = None, None

        return a, b

    def get_grounding_bb_qwen25(self, start, end) -> tuple:
        """
        Qwen2.5 works the same way, except the coordinates are no longer normalized and the format is:
        "(x0,y0,x1,y1)"
        """

        if start >= end:
            return None, None

        # Including leading space
        if start > 0 and self.current_description[start - 1] == " ":
            start -= 1

        # Repeat the same prompt up to the selection, with grounding tokens added
        prompt = self.get_prompt()
        prompt += self.current_description[:start]
        prompt += "<|object_ref_start|>"
        prompt += self.current_description[start:end]
        prompt += "<|object_ref_end|><|box_start|>("

        bb_string, res = self.generator.generate(
            prompt = prompt,
            add_bos = True,
            max_new_tokens = 28,
            stop_conditions = [self.tokenizer.single_id("<|box_end|>")],
            gen_settings = ExLlamaV2Sampler.Settings.greedy(),
            embeddings = [self.current_emb],
            completion_only = True,
            return_last_results = True,  # debug purposes
        )
        bb_string = "(" + bb_string

        print(f"Generation: {bb_string}")
        pprint.pprint(res, indent = 4)

        # BB string is in the format "(x0,y0,x1,y1)" with integer coordinates

        s = self.current_image.size
        try:
            d = tuple(map(int, bb_string.strip("()").split(",")))
            a = (d[0] / s[0], d[1] / s[1])
            b = (d[2] / s[0], d[3] / s[1])
        except:
            print("No bounding box could be determined")
            a, b = None, None

        return a, b



class GroundingDemo(QMainWindow):

    model: Model

    class CustomTextEdit(QTextEdit):
        """Custom QTextEdit that emits a signal when a selection is completed."""
        selection_complete = pyqtSignal(tuple)

        def mouseReleaseEvent(self, event):
            """Handle mouse release and emit the selection complete signal."""
            super().mouseReleaseEvent(event)
            cursor = self.textCursor()

            if cursor.hasSelection():
                # Start with the selected range
                start = cursor.selectionStart()
                end = cursor.selectionEnd()

                # Move to the start of the selection and expand to the start of the word
                cursor.setPosition(start)
                cursor.movePosition(QTextCursor.StartOfWord, QTextCursor.MoveAnchor)
                expanded_start = cursor.position()

                # Move to the end of the selection and expand to the end of the word
                cursor.setPosition(end)
                cursor.movePosition(QTextCursor.EndOfWord, QTextCursor.MoveAnchor)
                expanded_end = cursor.position()

                # Update the selection
                cursor.setPosition(expanded_start, QTextCursor.MoveAnchor)
                cursor.setPosition(expanded_end, QTextCursor.KeepAnchor)
                self.setTextCursor(cursor)  # Update the visible selection

                # Emit the expanded selection range
                self.selection_complete.emit((expanded_start, expanded_end))

    class CustomQLabel(QLabel):
        def __init__(self, parent, callback):
            super().__init__(parent)
            self.setAcceptDrops(True)
            self.callback = callback
            self.bounding_box = None
            self.scale = (1, 1)

        def setEnabled(self, enabled):
            """Override setEnabled to prevent grayscaling."""
            super().setEnabled(True)

        def set_bounding_box(self, a, b):
            """Set the bounding box to be drawn."""
            if a is None:
                self.clear_bounding_box()
                return
            inner_rect = self.contentsRect()
            w, h = inner_rect.width(), inner_rect.height()
            iw, ih = self.scale
            x1, y1 = a
            x2, y2 = b
            x1 = int(x1 * iw + (w - iw) / 2)
            y1 = int(y1 * ih + (h - ih) / 2)
            x2 = int(x2 * iw + (w - iw) / 2)
            y2 = int(y2 * ih + (h - ih) / 2)
            self.bounding_box = QRect(x1, y1, x2 - x1, y2 - y1)
            self.update()

        def clear_bounding_box(self):
            """Clear the bounding box."""
            self.bounding_box = None
            self.update()

        def paintEvent(self, event):
            """Override paintEvent to draw the bounding box."""
            super().paintEvent(event)

            if self.bounding_box:
                painter = QPainter(self)
                overlay_color = QColor(64, 64, 64, 150)
                painter.setBrush(overlay_color)
                painter.setPen(Qt.NoPen)
                full_region = QRegion(self.rect())
                exclude_region = QRegion(self.bounding_box)
                clip_region = full_region.subtracted(exclude_region)
                painter.setClipRegion(clip_region)
                painter.drawRect(self.rect())
                pen = QPen(Qt.white, 2)
                painter.setPen(pen)
                painter.setBrush(overlay_color)
                painter.drawRect(self.bounding_box)

        def dragEnterEvent(self, event):
            """Handle drag enter events."""
            if event.mimeData().hasUrls():
                event.accept()
            else:
                event.ignore()

        def dropEvent(self, event):
            """Handle drop events."""
            if event.mimeData().hasUrls():
                # Get the first file path or URL
                urls = event.mimeData().urls()
                url = urls[0]
                file_path_or_url = url.toString()
                if not file_path_or_url.startswith(("http://", "https://")):
                    file_path_or_url = url.toLocalFile()
                self.callback(file_path_or_url)  # Pass the local file path to the callback
            else:
                event.ignore()

    def __init__(self, model: Model, title: str):
        super().__init__()
        self.model = model
        self.no_events_plz = False

        self.setWindowTitle(f"Grounding Demo - {title}")
        self.setGeometry(100, 100, 1000, 1000)

        font = QFont()
        font.setPointSize(11)  # Set the font size to 16 points

        # Main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)

        # Image display
        self.image_label = self.CustomQLabel(self, self.load_dropped_image)
        self.image_label.setText("Image goes here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #404040; color: white;")
        self.image_label.setFont(font)
        main_layout.addWidget(self.image_label, stretch = 5)

        # Button row
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)

        self.paste_button = QPushButton("Paste Image")
        self.paste_button.clicked.connect(self.paste_image)
        self.paste_button.setFont(font)
        button_layout.addWidget(self.paste_button)

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        self.load_button.setFont(font)
        button_layout.addWidget(self.load_button)

        self.inference_button = QPushButton("Inference")
        self.inference_button.clicked.connect(self.run_inference)
        self.inference_button.setFont(font)
        button_layout.addWidget(self.inference_button)

        # Model output
        self.output_label = QLabel("Model Output:", self)
        self.output_label.setStyleSheet("color: white;")
        self.output_label.setFont(font)
        main_layout.addWidget(self.output_label)

        self.output_text = self.CustomTextEdit(self)
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("background-color: #3C3C3C; color: white;")
        self.output_text.setFont(font)
        main_layout.addWidget(self.output_text, stretch = 2)

        self.output_text.selection_complete.connect(self.on_selection_made)
        self.previous_selection = ""

        # Set dark theme
        self.setStyleSheet("background-color: #2E2E2E; color: white;")

    def load_dropped_image(self, file_path_or_url):
        """Load an image when it is dropped on the image label."""
        try:
            if file_path_or_url.startswith(("http://", "https://")):
                # Handle web URL
                response = requests.get(file_path_or_url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(file_path_or_url)
            self.display_image(image, from_pil=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def paste_image(self):
        """Paste an image from the clipboard."""
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()
        if mime_data.hasImage():
            qt_image = clipboard.image()
            self.display_image(qt_image, from_pil = False)
        else:
            QMessageBox.warning(self, "Error", "No image found in clipboard.")

    def load_image(self):
        """Open a file dialog to load an image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
        )
        if file_path:
            try:
                image = Image.open(file_path)
                self.display_image(image, from_pil = True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def run_inference(self):
        try:
            self.no_events_plz = True
            self.paste_button.setEnabled(False)
            self.load_button.setEnabled(False)
            self.inference_button.setEnabled(False)
            self.output_text.setText("")
            self.model.inference(self.output_text.setText, QApplication.processEvents)
        finally:
            self.no_events_plz = False
            self.paste_button.setEnabled(True)
            self.load_button.setEnabled(True)
            self.inference_button.setEnabled(True)

    def display_image(self, image, from_pil):
        # Convert and display the image
        if self.no_events_plz:
            return

        if from_pil:
            self.model.set_image(image)
            image = image.convert("RGBA")
            data = image.tobytes("raw", "RGBA")
            q_image = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
        else:
            # If the image comes from a QImage (e.g., clipboard), convert to PIL
            self.model.set_image(self.qimage_to_pil(image))
            q_image = image

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scaled_width = scaled_pixmap.width()
        scaled_height = scaled_pixmap.height()
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.clear_bounding_box()
        self.image_label.scale = (scaled_width, scaled_height)
        self.output_text.setText("")

    def qimage_to_pil(self, q_image):
        """Convert a QImage to a PIL Image."""
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        q_image.save(buffer, "PNG")
        pil_image = Image.open(BytesIO(buffer.data()))
        return pil_image

    def on_selection_made(self, pos):
        """Callback for when a selection is made."""
        if self.no_events_plz:
            return

        start, end = pos
        # start, end = model.expand_selection(start, end)

        print(f"Selected span: {start}, {end}")
        print(f"Selected text: {repr(self.model.current_description[start:end])}")
        a, b = self.model.bbox_func(start, end)
        self.image_label.set_bounding_box(a, b)


# Qwen2-VL 7B:
#   https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
#   https://huggingface.co/turboderp/Qwen2-VL-7B-Instruct-exl2

def main():

    # model_dir = "/mnt/str/models/qwen2-vl-7b-instruct-exl2/6.0bpw"
    # bbox_mode = "qwen25"
    model_dir = "/mnt/str/models/qwen2.5-vl-7b-instruct-exl2/6.0bpw"
    bbox_mode = "qwen25"

    app = QApplication(sys.argv)
    model = Model(model_dir, bbox_mode)
    model.load()
    window = GroundingDemo(model, model_dir)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
