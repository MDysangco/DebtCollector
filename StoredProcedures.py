from db_loader import get_engine
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

def insert_configuration(
    BuyProbabilityThreshold,
    SellProbabilityThreshold,
    TrendEMALength,
    VolFilterWindow,
    VolMinThreshold,
    GlobalThreshold,
    PerSymbolFloor,
    Margin,
    CooldownHours
):
    engine = get_engine()

    sql = text("""
        DECLARE @NewId INT;

        EXEC InsertConfiguration
            @BuyProbabilityThreshold = :BuyProbabilityThreshold,
            @SellProbabilityThreshold = :SellProbabilityThreshold,
            @TrendEMALength = :TrendEMALength,
            @VolFilterWindow = :VolFilterWindow,
            @VolMinThreshold = :VolMinThreshold,
            @GlobalThreshold = :GlobalThreshold,
            @PerSymbolFloor = :PerSymbolFloor,
            @Margin = :Margin,
            @CooldownHours = :CooldownHours,
            @NewId = @NewId OUTPUT;

        SELECT @NewId AS ConfigRowId;
    """)

    with engine.begin() as conn:
        result = conn.execute(sql, {
            "BuyProbabilityThreshold": BuyProbabilityThreshold,
            "SellProbabilityThreshold": SellProbabilityThreshold,
            "TrendEMALength": TrendEMALength,
            "VolFilterWindow": VolFilterWindow,
            "VolMinThreshold": VolMinThreshold,
            "GlobalThreshold": GlobalThreshold,
            "PerSymbolFloor": PerSymbolFloor,
            "Margin": Margin,
            "CooldownHours": CooldownHours
        })

        row = result.fetchone()
        return int(row[0])

def insert_raw_reading(
        TimestampUtc,
        CoinId,
        PredictedClass,
        ProbSell,
        ProbHold,
        ProbBuy,
        Price,
        EMA,
        Volatility,
        PassedProbFilter,
        PassedTrendFilter,
        PassedVolFilter,
        FinalSignal,
        ModelId,
        ConfigRowId
):
    """
    Calls the InsertRawReading stored procedure.
    Duplicate key errors are ignored (due to UNIQUE index).
    """

    engine = get_engine()

    sql = text("""
        EXEC InsertRawReading
            @TimestampUtc = :TimestampUtc,
            @CoinId = :CoinId,
            @PredictedClass = :PredictedClass,
            @ProbSell = :ProbSell,
            @ProbHold = :ProbHold,
            @ProbBuy = :ProbBuy,
            @Price = :Price,
            @EMA = :EMA,
            @Volatility = :Volatility,
            @PassedProbFilter = :PassedProbFilter,
            @PassedTrendFilter = :PassedTrendFilter,
            @PassedVolFilter = :PassedVolFilter,
            @FinalSignal = :FinalSignal,
            @ModelId = :ModelId,
            @ConfigRowId = :ConfigRowId;
    """)

    try:
        with engine.begin() as conn:
            conn.execute(sql, {
                "TimestampUtc": TimestampUtc,
                "CoinId": CoinId,
                "PredictedClass": PredictedClass,
                "ProbSell": ProbSell,
                "ProbHold": ProbHold,
                "ProbBuy": ProbBuy,
                "Price": Price,
                "EMA": EMA,
                "Volatility": Volatility,
                "PassedProbFilter": PassedProbFilter,
                "PassedTrendFilter": PassedTrendFilter,
                "PassedVolFilter": PassedVolFilter,
                "FinalSignal": FinalSignal,
                "ModelId": ModelId,
                "ConfigRowId": ConfigRowId
            })
    except IntegrityError:
        # Duplicate key (TimestampUtc, CoinId, ModelId)
        print(f"Duplicate RawReading skipped: {TimestampUtc} CoinId={CoinId} ModelId={ModelId}")
