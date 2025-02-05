import json
import logging
import warnings

import trio
import urwid


class WorkerMonitor:
    def __init__(self):
        self.requests = []
        self.header_info = {}

        self.palette = [
            ('headerbar',         'white',      'dark blue'),
            ('request_row',       'white',      'dark gray'),
            ('worker_row',        'light gray', 'black'),
            ('progress_normal',   'black',      'light gray'),
            ('progress_complete', 'black',      'dark green'),
            ('body',              'white',      'black'),
        ]

        # --- Top bar (header) ---
        worker_name = self.header_info.get('left', "unknown")
        balance     = self.header_info.get('right', "balance: unknown")

        self.worker_name_widget = urwid.Text(worker_name)
        self.balance_widget     = urwid.Text(balance, align='right')

        header = urwid.Columns([self.worker_name_widget, self.balance_widget])
        header_attr = urwid.AttrMap(header, 'headerbar')

        # --- Body (List of requests) ---
        self.body_listbox = self._create_listbox_body(self.requests)

        # --- Bottom bar (progress) ---
        self.status_text  = urwid.Text("Request: none", align='left')
        self.progress_bar = urwid.ProgressBar(
            'progress_normal',
            'progress_complete',
            current=0,
            done=100
        )

        footer_cols = urwid.Columns([
            ('fixed', 20, self.status_text),
            self.progress_bar,
        ])

        # Build the main frame
        frame = urwid.Frame(
            self.body_listbox,
            header=header_attr,
            footer=footer_cols
        )

        # Set up the main loop with Trio
        self.event_loop = urwid.TrioEventLoop()
        self.main_loop = urwid.MainLoop(
            frame,
            palette=self.palette,
            event_loop=self.event_loop,
            unhandled_input=self._exit_on_q
        )

    def _create_listbox_body(self, requests):
        """
        Build a ListBox (vertical list) of requests & workers using SimpleFocusListWalker.
        """
        widgets = self._build_request_widgets(requests)
        walker = urwid.SimpleFocusListWalker(widgets)
        return urwid.ListBox(walker)

    def _build_request_widgets(self, requests):
        """
        Build a list of Urwid widgets (one row per request + per worker).
        """
        row_widgets = []

        for req in requests:
            # Build a columns widget for the request row
            columns = urwid.Columns([
                ('fixed', 5,  urwid.Text(f"#{req['id']}")),   # e.g. "#12"
                ('weight', 3, urwid.Text(req['model'])),
                ('weight', 3, urwid.Text(req['prompt'])),
                ('fixed', 13, urwid.Text(req['user'])),
                ('fixed', 13, urwid.Text(req['reward'])),
            ], dividechars=1)

            # Wrap the columns with an attribute map for coloring
            request_row = urwid.AttrMap(columns, 'request_row')
            row_widgets.append(request_row)

            # Then add each worker in its own line below
            for w in req["workers"]:
                worker_line = urwid.Text(f"  {w}")
                worker_row  = urwid.AttrMap(worker_line, 'worker_row')
                row_widgets.append(worker_row)

            # Optional blank line after each request
            row_widgets.append(urwid.Text(""))

        return row_widgets

    def _exit_on_q(self, key):
        """Exit the TUI on 'q' or 'Q'."""
        if key in ('q', 'Q'):
            raise urwid.ExitMainLoop()

    async def run(self):
        """
        Run the TUI in an async context (Trio).
        This method blocks until the user quits (pressing q/Q).
        """
        with self.main_loop.start():
            await self.event_loop.run_async()

        raise urwid.ExitMainLoop()

    # -------------------------------------------------------------------------
    # Public Methods to Update Various Parts of the UI
    # -------------------------------------------------------------------------
    def set_status(self, status: str):
        self.status_text.set_text(status)

    def set_progress(self, current, done=None):
        """
        Update the bottom progress bar.
          - `current`: new current progress value (int).
          - `done`: max progress value (int). If None, we don’t change it.
        """
        if done is not None:
            self.progress_bar.done = done

        self.progress_bar.current = current

        pct = 0
        if self.progress_bar.done != 0:
            pct = int((self.progress_bar.current / self.progress_bar.done) * 100)

    def update_requests(self, new_requests):
        """
        Replace the data in the existing ListBox with new request widgets.
        """
        new_widgets = self._build_request_widgets(new_requests)
        self.body_listbox.body[:] = new_widgets  # replace content of the list walker

    def set_header_text(self, new_worker_name=None, new_balance=None):
        """
        Update the text in the header bar for worker name and/or balance.
        """
        if new_worker_name is not None:
            self.worker_name_widget.set_text(new_worker_name)
        if new_balance is not None:
            self.balance_widget.set_text(new_balance)

    def network_update(self, snapshot: dict):
        queue = [
            {
                **r,
                **(json.loads(r['body'])['params']),
                'workers': [s['worker'] for s in snapshot['requests'][r['id']]]
            }
            for r in snapshot['queue']
        ]
        self.update_requests(queue)


def setup_logging_for_tui(level):
    warnings.filterwarnings("ignore")

    logger = logging.getLogger()
    logger.setLevel(level)

    fh = logging.FileHandler('dgpu.log')
    fh.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)


_tui = None
def init_tui():
    global _tui
    assert not _tui
    setup_logging_for_tui(logging.INFO)
    _tui = WorkerMonitor()
    return _tui


def maybe_update_tui(fn):
    global _tui
    if _tui:
        fn(_tui)


async def maybe_update_tui_async(fn):
    global _tui
    if _tui:
        await fn(_tui)
