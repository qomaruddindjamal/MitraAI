FROM rust:1.82-bookworm AS builder

WORKDIR /usr/src/mitra_ai
COPY . .

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/src/mitra_ai/target/release/mitra_ai /app/mitra_ai
COPY models /app/models

ENV PORT=8080
ENV HOST=0.0.0.0

EXPOSE 8080

CMD ["/app/mitra_ai", "--serve"]
