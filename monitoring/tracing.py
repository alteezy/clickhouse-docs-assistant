import json
from datetime import datetime, timezone

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from db import connect, init_db


def _ns_to_dt(ns):
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def _extract_question(attrs):
    """The user's question isn't its own attribute - it's embedded as the
    first user message inside pydantic-ai's full-conversation JSON dump."""
    raw = attrs.get("pydantic_ai.all_messages")
    if not raw:
        return None
    try:
        messages = json.loads(raw)
        for msg in messages:
            if msg.get("role") == "user":
                for part in msg.get("parts", []):
                    if part.get("type") == "text":
                        return part.get("content")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    return None


class PostgresSpanExporter(SpanExporter):
    """Writes finished OTel spans to Postgres.

    Every span (LLM calls, tool calls, the whole agent run) is persisted with
    its raw attributes as JSONB for flexible querying. The root "invoke_agent"
    span additionally gets question/answer/token counts pulled out into their
    own columns, since that's what the dashboard's conversation-level charts
    need most often.
    """

    def export(self, spans):
        try:
            con = connect()
        except Exception:
            return SpanExportResult.FAILURE
        try:
            with con, con.cursor() as cur:
                for span in spans:
                    attrs = dict(span.attributes or {})
                    is_root = span.name.startswith("invoke_agent")
                    cur.execute(
                        """
                        INSERT INTO spans
                            (trace_id, span_id, parent_span_id, name, start_time, end_time,
                             duration_ms, question, answer, input_tokens, output_tokens, attributes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (trace_id, span_id) DO NOTHING
                        """,
                        (
                            format(span.context.trace_id, "032x"),
                            format(span.context.span_id, "016x"),
                            format(span.parent.span_id, "016x") if span.parent else None,
                            span.name,
                            _ns_to_dt(span.start_time),
                            _ns_to_dt(span.end_time),
                            (span.end_time - span.start_time) / 1e6,
                            _extract_question(attrs) if is_root else None,
                            attrs.get("final_result") if is_root else None,
                            attrs.get("gen_ai.aggregated_usage.input_tokens") if is_root else None,
                            attrs.get("gen_ai.aggregated_usage.output_tokens") if is_root else None,
                            json.dumps(attrs, default=str),
                        ),
                    )
        except Exception:
            return SpanExportResult.FAILURE
        finally:
            con.close()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True


def build_tracer_provider():
    try:
        init_db()
    except Exception as e:
        print(f"monitoring: could not reach Postgres to init schema ({e}) - tracing will no-op until it's up")

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(PostgresSpanExporter()))
    return provider
