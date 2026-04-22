"""Configuration for the bias scanner."""

from dataclasses import dataclass, field


@dataclass
class ScannerConfig:
    """
    Configuration for the Polymarket bias scanner.

    Attributes:
        min_volume: Minimum trading volume in USD
        min_liquidity: Minimum liquidity in USD
        max_days_to_expiry: Maximum days until resolution
        llm_model: LLM model for classification
        max_markets: Maximum markets to fetch
        output_dir: Directory for output reports
        verbose: Enable verbose logging
        always_include_keywords: Markets matching any of these keywords bypass all
            other filters (volume, liquidity, geography, etc.).
    """
    min_volume: float = 5000
    min_liquidity: float = 2000
    max_days_to_expiry: int = 90
    llm_model: str = "claude-haiku-4-5"
    max_markets: int = 500
    max_reported_markets: int = 20
    output_dir: str = "output"
    verbose: bool = False
    always_include_keywords: list[str] = field(default_factory=lambda: [
        # PSOE — historic & current leadership
        "Pedro Sanchez", "Zapatero", "Felipe Gonzalez",
        "Alfredo Perez Rubalcaba", "Joaquin Almunia", "Patxi Lopez",
        "Carmen Calvo", "Maria Jesus Montero", "Teresa Ribera", "Adriana Lastra",
        "Abalos", "Koldo",
        # Podemos / Sumar / IU
        "Pablo Iglesias", "Yolanda Diaz", "Irene Montero",
        "Ione Belarra", "Alberto Garzon",
        # Regional / municipal left
        "Ada Colau", "Monica Oltra",
        # UFO / UAP / extraterrestrial
        "UAP", "UFO", "extraterrestrial life", "alien life", "alien civilization",
        "alien contact", "alien disclosure", "Area 51", "Roswell", "first contact",
        # Religious prophecy & end-times
        "Second Coming", "return of Jesus", "return of Christ",
        "Rapture", "End Times", "Antichrist", "Armageddon", "biblical prophecy",
        "apocalypse",
    ])
