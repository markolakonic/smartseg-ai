"""
Učitavanje i čišćenje podataka — ROBUSTNO.

Prima CSV ili Excel u raznim formatima i vraća čist DataFrame na nivou
transakcionih linija. Prepoznaje različite nazive kolona, separatore (`,` `;`
tab), encoding-zi, i izvodi nedostajuće kolone gdje je moguće (npr. ako nema
InvoiceNo, sintetiše ga; ako ima samo Amount umjesto Quantity*UnitPrice).
"""
from __future__ import annotations

import io
import re
import pandas as pd

# Tačni nazivi -> standardni naziv
COLUMN_ALIASES = {
    # narudžba / faktura
    "invoiceno": "InvoiceNo", "invoice": "InvoiceNo", "invoice_no": "InvoiceNo",
    "invoice no": "InvoiceNo", "orderid": "InvoiceNo", "order id": "InvoiceNo",
    "order_id": "InvoiceNo", "order": "InvoiceNo", "ordernumber": "InvoiceNo",
    "transactionid": "InvoiceNo", "transaction_id": "InvoiceNo", "transaction": "InvoiceNo",
    "billno": "InvoiceNo", "bill_no": "InvoiceNo", "receipt": "InvoiceNo",
    # proizvod
    "stockcode": "StockCode", "stock_code": "StockCode", "sku": "StockCode",
    "productid": "StockCode", "product_id": "StockCode", "itemid": "StockCode",
    "item_id": "StockCode", "itemcode": "StockCode",
    # opis
    "description": "Description", "product": "Description", "productname": "Description",
    "product_name": "Description", "item": "Description", "itemname": "Description",
    # količina
    "quantity": "Quantity", "qty": "Quantity", "units": "Quantity", "count": "Quantity",
    "unitssold": "Quantity", "units_sold": "Quantity",
    # datum
    "invoicedate": "InvoiceDate", "date": "InvoiceDate", "orderdate": "InvoiceDate",
    "order_date": "InvoiceDate", "transactiondate": "InvoiceDate", "purchasedate": "InvoiceDate",
    "purchase_date": "InvoiceDate", "datetime": "InvoiceDate", "timestamp": "InvoiceDate",
    "order date": "InvoiceDate",
    # cijena po jedinici
    "unitprice": "UnitPrice", "price": "UnitPrice", "unit_price": "UnitPrice",
    "unit price": "UnitPrice", "priceperunit": "UnitPrice", "price_per_unit": "UnitPrice",
    # kupac
    "customerid": "CustomerID", "customer id": "CustomerID", "custid": "CustomerID",
    "customer_id": "CustomerID", "customer": "CustomerID", "userid": "CustomerID",
    "user_id": "CustomerID", "clientid": "CustomerID", "client_id": "CustomerID",
    "cust_id": "CustomerID", "client": "CustomerID", "user": "CustomerID",
    # zemlja
    "country": "Country", "nation": "Country", "region": "Country", "market": "Country",
    # ukupan iznos (direktna monetarna kolona)
    "totalprice": "TotalPrice", "total": "TotalPrice", "amount": "TotalPrice",
    "totalamount": "TotalPrice", "total_amount": "TotalPrice", "sales": "TotalPrice",
    "revenue": "TotalPrice", "total_price": "TotalPrice", "linetotal": "TotalPrice",
    "line_total": "TotalPrice", "subtotal": "TotalPrice", "spend": "TotalPrice",
    "total price": "TotalPrice", "total amount": "TotalPrice",
    "finalprice": "TotalPrice", "final_price": "TotalPrice", "final price": "TotalPrice",
    "netprice": "TotalPrice", "net_price": "TotalPrice", "amountpaid": "TotalPrice",
    "amount_paid": "TotalPrice", "paidamount": "TotalPrice", "grandtotal": "TotalPrice",
    "grand_total": "TotalPrice", "purchaseamount": "TotalPrice", "purchase_amount": "TotalPrice",
    "ordervalue": "TotalPrice", "order_value": "TotalPrice", "ordertotal": "TotalPrice",
}


def _norm(col) -> str:
    """Normalizuj naziv kolone: izbaci (jedinice), %, razmake -> 'Final_Price(Rs.)' => 'final_price'."""
    c = str(col).strip().lower()
    c = re.sub(r"\(.*?\)", "", c)                 # ukloni (Rs.), (%) itd.
    c = re.sub(r"[^a-z0-9]+", "_", c).strip("_")   # ujednači razmake/znakove
    return c


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tačno mapiranje (preko normalizovanog naziva) + fuzzy za nemapirane kolone."""
    rename = {}
    for col in df.columns:
        key = _norm(col)
        if key in COLUMN_ALIASES:
            rename[col] = COLUMN_ALIASES[key]
        elif str(col).strip().lower() in COLUMN_ALIASES:
            rename[col] = COLUMN_ALIASES[str(col).strip().lower()]
    df = df.rename(columns=rename)

    std = {"InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
           "UnitPrice", "CustomerID", "Country", "TotalPrice"}
    have = set(df.columns)
    for col in list(df.columns):
        if col in std:
            continue
        low = _norm(col)
        target = None
        # ukupan iznos ima prioritet nad jediničnom cijenom (final/net/paid/grand + price/amount)
        if (("final" in low or "net" in low or "paid" in low or "grand" in low or "order" in low)
                and ("price" in low or "amount" in low or "total" in low or "value" in low)
                and "TotalPrice" not in have):
            target = "TotalPrice"
        elif ("amount" in low or "total" in low or "revenue" in low or "sales" in low or "spend" in low) and "TotalPrice" not in have:
            target = "TotalPrice"
        elif ("custom" in low or "client" in low or "user" in low or "buyer" in low) and "CustomerID" not in have:
            target = "CustomerID"
        elif ("date" in low or "time" in low) and "InvoiceDate" not in have:
            target = "InvoiceDate"
        elif ("qty" in low or "quant" in low or "units" in low) and "Quantity" not in have:
            target = "Quantity"
        elif "price" in low and "UnitPrice" not in have:
            target = "UnitPrice"
        elif ("invoice" in low or "order" in low or "transac" in low or "bill" in low or "receipt" in low) and "InvoiceNo" not in have:
            target = "InvoiceNo"
        elif ("countr" in low or "nation" in low) and "Country" not in have:
            target = "Country"
        if target:
            df = df.rename(columns={col: target})
            have.add(target)
    return df


def read_any(file_or_bytes, filename: str | None = None) -> pd.DataFrame:
    """Učitaj CSV ili Excel — pokušava više encoding-a i auto-detekciju separatora."""
    name = (filename or getattr(file_or_bytes, "name", "") or "").lower()
    if name.endswith((".xlsx", ".xls")):
        src = io.BytesIO(file_or_bytes) if isinstance(file_or_bytes, (bytes, bytearray)) else file_or_bytes
        try:                                   # calamine je ~4-5x brži za velike fajlove
            return pd.read_excel(src, engine="calamine")
        except Exception:
            if hasattr(src, "seek"):
                src.seek(0)
            return pd.read_excel(src)

    raw = file_or_bytes if isinstance(file_or_bytes, (bytes, bytearray)) else None

    def _try(enc, sep):
        buf = io.BytesIO(raw) if raw is not None else file_or_bytes
        return pd.read_csv(buf, encoding=enc, sep=sep, engine="python" if sep is None else "c")

    for enc in ("utf-8-sig", "utf-8", "ISO-8859-1", "latin1", "cp1252"):
        try:
            df = _try(enc, ",")
            if df.shape[1] == 1:                 # vjerovatno pogrešan separator
                df = _try(enc, None)             # auto-detekcija (; tab ...)
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    # zadnji pokušaj
    return pd.read_csv(io.BytesIO(raw) if raw is not None else file_or_bytes,
                       encoding="ISO-8859-1", sep=None, engine="python")


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizuj, izvedi nedostajuće kolone i očisti podatke."""
    df = _standardize_columns(df).copy()

    # Minimum: kupac + datum + monetarno (TotalPrice ILI Quantity*UnitPrice)
    if "CustomerID" not in df.columns:
        raise ValueError("Ne mogu pronaći kolonu sa ID-em kupca "
                         "(CustomerID / Customer / UserID / ClientID...).")
    if "InvoiceDate" not in df.columns:
        raise ValueError("Ne mogu pronaći kolonu sa datumom "
                         "(InvoiceDate / Date / OrderDate / Timestamp...).")
    has_total = "TotalPrice" in df.columns
    has_qp = "Quantity" in df.columns and "UnitPrice" in df.columns
    if not (has_total or has_qp):
        raise ValueError("Ne mogu pronaći monetarnu kolonu "
                         "(Amount / Total / Sales / Revenue, ili UnitPrice + Quantity).")

    # Kupac (broj ili tekst)
    df = df.dropna(subset=["CustomerID"])
    try:
        df["CustomerID"] = df["CustomerID"].astype(float).astype("int64")
    except (ValueError, TypeError):
        df["CustomerID"] = df["CustomerID"].astype(str).str.strip()

    # Datum
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # Faktura — sintetiši ako fali (jedna narudžba = kupac + dan)
    _synth_invoice = "InvoiceNo" not in df.columns
    if _synth_invoice:
        df["InvoiceNo"] = ("ORD-" + df["CustomerID"].astype(str) + "-"
                           + df["InvoiceDate"].dt.strftime("%Y%m%d"))
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    if not _synth_invoice:                        # storna se filtriraju samo za PRAVE fakture
        df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]

    # Količina (default 1 ako fali)
    if "Quantity" not in df.columns:
        df["Quantity"] = 1
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(1)

    # Monetarno
    if has_total:
        df["TotalPrice"] = pd.to_numeric(df["TotalPrice"], errors="coerce")
        if "UnitPrice" not in df.columns:
            df["UnitPrice"] = df["TotalPrice"] / df["Quantity"].replace(0, 1)
    else:
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Zadrži samo validne (pozitivne) transakcije
    df = df[(df["Quantity"] > 0) & (df["TotalPrice"] > 0)]

    # Dodatne kolone sa defaultima
    if "Country" not in df.columns:
        df["Country"] = "Unknown"
    if "StockCode" not in df.columns:
        df["StockCode"] = df["Description"] if "Description" in df.columns else "ITEM"

    df = df.drop_duplicates()                     # ukloni duplikate

    if len(df) == 0:
        raise ValueError("Nakon čišćenja nema validnih redova. Provjeri da fajl ima "
                         "pozitivne količine/iznose i ispravne datume.")
    return df.reset_index(drop=True)


def dataset_summary(df: pd.DataFrame) -> dict:
    """Brzi pregled za prikaz korisniku nakon učitavanja."""
    return {
        "n_transactions": int(df["InvoiceNo"].nunique()),
        "n_lines": int(len(df)),
        "n_customers": int(df["CustomerID"].nunique()),
        "n_countries": int(df["Country"].nunique()),
        "total_revenue": float(df["TotalPrice"].sum()),
        "date_min": df["InvoiceDate"].min(),
        "date_max": df["InvoiceDate"].max(),
    }
