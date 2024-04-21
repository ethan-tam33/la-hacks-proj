from client.templates import template
import reflex as rx

class AnalyzeState(rx.State):
    is_healthy: str
    gemini: str
    nerf_url: str
    file_name: str
    confidence_score: str

    async def update_is_healthy(self, new_is_healthy):
        self.is_healthy = new_is_healthy
        
    async def update_gemini(self, new_gemini):
        self.gemini = new_gemini
    
    async def update_nerf_url(self, new_nerf_url):
        self.nerf_url = new_nerf_url
    
    async def update_file_name(self, new_file_name):
        self.file_name = new_file_name

    async def update_confidence_score(self, new_confidence_score):
        self.confidence_score = new_confidence_score
    
@template(route="/analyze", title="Analyze")
def analyze() -> rx.Component:
    return rx.vstack(
        rx.image(
            src="../Reefer.png",
            width="100px"
        ),
        rx.link(
            rx.button(
                "Back",
            ),
            href="/"
        ),
        rx.heading("Results", size="8"),
        rx.text(AnalyzeState.is_healthy),
        rx.text(AnalyzeState.confidence_score),
        rx.text(AnalyzeState.gemini),
        # rx.image(
        #     src=AnalyzeState.file_name,
        #     border="5px solid #555"
        # ),
        rx.video(
            url=AnalyzeState.nerf_url,
            width="400px",
            height="auto",
            playing=True,
            loop=True
        ),
    )
