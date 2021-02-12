from .world import World
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import subprocess
styles = getSampleStyleSheet()


class Report:
    def __init__(self, world: World):
        self.world = world
        self.filepath = "report.pdf"
        self.doc = SimpleDocTemplate(self.filepath)
        self.title = "MonteCarlo Experiment Report"
        self.info = "Experiment Report"
        self.pageWidth = 612
        self.pageHeight = 792
        self.canvas = Canvas(self.filepath, pagesize=letter)
        self.canvas.drawString(0, 0, "Hello World!")

    def myFirstPage(self, canvas, doc):
        canvas.saveState()
        canvas.setFont('Times-Bold', 16)
        canvas.drawCentredString(self.pageWidth / 2.0, self.pageHeight - 108, self.title)
        canvas.restoreState()

    def myLaterPages(self, canvas, doc):
        canvas.saveState()
        canvas.setFont('Times-Roman', 9)
        canvas.drawString(inch, 0.75 * inch, "Page %d - %s" % (doc.page, "MonteCarlo Experiment Report"))
        canvas.restoreState()

    def generateFigures(self):
        pass

    def showReport(self):
        pass

    def generateReport(self):
        Story = [Spacer(1, 2 * inch)]
        style = styles["Normal"]
        for i in range(100):
            bogustext = ("This is Paragraph number %s. " % i) * 20
            p = Paragraph(bogustext, style)
            Story.append(p)
            Story.append(Spacer(1, 0.2 * inch))
        self.doc.build(Story, onFirstPage=self.myFirstPage, onLaterPages=self.myLaterPages)

    def show(self):
        subprocess.Popen(["report.pdf"], shell=True)
